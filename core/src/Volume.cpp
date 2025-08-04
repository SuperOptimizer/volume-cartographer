#include "vc/core/types/Volume.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/metadata.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/util/util.hxx"
#include "z5/util/blocking.hxx"
#include "z5/util/format_data.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)

namespace fs = volcart::filesystem;

using namespace volcart;

// Load a Volume from disk
Volume::Volume(fs::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    width_ = metadata_.get<int>("width");
    height_ = metadata_.get<int>("height");
    slices_ = metadata_.get<int>("slices");
    numSliceCharacters_ = std::to_string(slices_).size();

    std::vector<std::mutex> init_mutexes(slices_);

    slice_mutexes_.swap(init_mutexes);
    
    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(fs::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name)),
          slice_mutexes_(slices_)
{
    metadata_.set("type", "vol");
    metadata_.set("width", width_);
    metadata_.set("height", height_);
    metadata_.set("slices", slices_);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});    

    zarrOpen();
}

void Volume::zarrOpen()
{
    if (!metadata_.hasKey("format") || metadata_.get<std::string>("format") != "zarr")
        return;

    isZarr = true;
    zarrFile_ = new z5::filesystem::handle::File(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);
    
    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::sort(groups.begin(), groups.end());
    
    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        z5::filesystem::handle::Dataset ds_handle(group, name, nlohmann::json::parse(std::ifstream(path_/name/".zarray")).value<std::string>("dimension_separator","."));

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);
    }
}

// Load a Volume from disk, return a pointer
auto Volume::New(fs::path path) -> Volume::Pointer
{
    return std::make_shared<Volume>(path);
}

// Set a Volume from a folder of slices, return a pointer
auto Volume::New(fs::path path, std::string uuid, std::string name)
    -> Volume::Pointer
{
    return std::make_shared<Volume>(path, uuid, name);
}

auto Volume::sliceWidth() const -> int { return width_; }
auto Volume::sliceHeight() const -> int { return height_; }
auto Volume::numSlices() const -> int { return slices_; }
auto Volume::voxelSize() const -> double
{
    return metadata_.get<double>("voxelsize");
}
auto Volume::min() const -> double { return metadata_.get<double>("min"); }
auto Volume::max() const -> double { return metadata_.get<double>("max"); }

void Volume::setSliceWidth(int w)
{
    width_ = w;
    metadata_.set("width", w);
}

void Volume::setSliceHeight(int h)
{
    height_ = h;
    metadata_.set("height", h);
}

void Volume::setNumberOfSlices(std::size_t numSlices)
{
    slices_ = numSlices;
    numSliceCharacters_ = std::to_string(numSlices).size();
    metadata_.set("slices", numSlices);
}

void Volume::setVoxelSize(double s) { metadata_.set("voxelsize", s); }
void Volume::setMin(double m) { metadata_.set("min", m); }
void Volume::setMax(double m) { metadata_.set("max", m); }

auto Volume::bounds() const -> Volume::Bounds
{
    return {
        {0, 0, 0},
        {static_cast<double>(width_), static_cast<double>(height_),
         static_cast<double>(slices_)}};
}

auto Volume::isInBounds(double x, double y, double z) const -> bool
{
    return x >= 0 && x < width_ && y >= 0 && y < height_ && z >= 0 &&
           z < slices_;
}

auto Volume::isInBounds(const cv::Vec3d& v) const -> bool
{
    return isInBounds(v(0), v(1), v(2));
}

void throw_run_path(const fs::path &path, const std::string msg)
{
    throw std::runtime_error(msg + " for " + path.string());
}

std::ostream& operator<< (std::ostream& out, const xt::xarray<uint8_t>::shape_type &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    return out;
}

z5::Dataset *Volume::zarrDataset(int level)
{
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales()
{
    return zarrDs_.size();
}

auto Volume::getSliceData(const int index) const -> cv::Mat
{
    if (cacheSlices_) {
        return cache_slice_(index);
    }
    // Never memory map if caching is disabled. This is mostly because there's
    // currently not a good way to clean up the mmap_info when not caching.
    return load_slice_(index);
}

auto Volume::getSliceDataCopy(const int index) const -> cv::Mat
{
    return getSliceData(index).clone();
}


auto Volume::cache_slice_(const int index) const -> cv::Mat
{
    // Check if the slice is in the cache.
    {
        std::shared_lock lock(cacheMutex_);
        if (cache_->contains(index)) {
            return cache_->get(index).first;
        }
    }

    {
        // If the slice is not in the cache, get exclusive access to this
        // slice's mutex. This slice can't be set until we're done.
        // TODO: Is this faster than just getting an unique cache lock?
        auto& mutex = sliceMutexes_[index];
        std::unique_lock lockSlice(mutex);

        // Check again to ensure the slice has not been added to the cache while
        // waiting for the lock.
        {
            std::shared_lock lockCache(cacheMutex_);
            if (cache_->contains(index)) {
                return cache_->get(index).first;
            }
        }

        // Load the slice and put it in the cache
        cv::Mat slice;
        std::optional<mmap_info> mmapInfo;
        // If memory mapping, get the mmap_info too
        if (memmap_) {
            mmap_info i;
            slice = load_slice_(index, &i);
            mmapInfo = i;
        } else {
            slice = load_slice_(index);
        }
        std::unique_lock lockCache(cacheMutex_);
        cache_->put(index, {slice, mmapInfo});
        return slice;
    }
}

void Volume::cachePurge() const
{
    std::unique_lock lock(cacheMutex_);
    cache_->purge();
}