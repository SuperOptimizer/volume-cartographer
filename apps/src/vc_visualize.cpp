#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/Slicing.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct SegmentData {
    cv::Mat_<cv::Vec3b> image;
    cv::Mat_<uint8_t> mask;
    cv::Vec2f offset;
    cv::Vec3b tint;
};

struct AlignmentResult {
    cv::Vec2f pixel_offset;
    int num_correspondences;
    bool valid;
};

class SegmentRenderer {
private:
    std::shared_ptr<VolumePkg> vpkg_;
    std::shared_ptr<Volume> volume_;
    ChunkCache* cache_;

    std::vector<cv::Vec3b> tint_colors_ = {
        cv::Vec3b(0, 100, 0),    // Green tint
        cv::Vec3b(100, 0, 0),    // Red tint
        cv::Vec3b(0, 0, 100),    // Blue tint
        cv::Vec3b(100, 100, 0),  // Yellow tint
        cv::Vec3b(100, 0, 100),  // Magenta tint
        cv::Vec3b(0, 100, 100),  // Cyan tint
    };

public:
    SegmentRenderer(const fs::path& volpkg_path, const std::string& volume_id) {
        vpkg_ = VolumePkg::New(volpkg_path.string());

        if (volume_id.empty()) {
            throw std::runtime_error("You must provide a volume id");
        }

        if (!vpkg_->hasVolume(volume_id)) {
            throw std::runtime_error("Volume not found: " + volume_id);
        }
        volume_ = vpkg_->volume(volume_id);
        std::cout << "Using volume: " << volume_id << " (" << volume_->name() << ")" << std::endl;
        std::cout << "Volume dimensions: " << volume_->sliceWidth() << "x" << volume_->sliceHeight() << "x" << volume_->numSlices() << std::endl;
        std::cout << "Available scales: " << volume_->numScales() << std::endl;
        cache_ = new ChunkCache(2ULL * 1024ULL * 1024ULL * 1024ULL);
    }

    ~SegmentRenderer() {
        delete cache_;
    }

    cv::Mat render(const std::string& segment_id, const fs::path& output_path,
                   std::string source, float opacity = 0.4) {

        // Special handling for sequence source
        if (source == "sequence") {
            return renderSequence(segment_id, output_path, opacity);
        }

        // Original behavior for other sources
        std::cout << "Rendering segment: " << segment_id << " with overlaps" << std::endl;

        // Load root segment
        auto root_meta = vpkg_->loadSurface(segment_id);
        if (!root_meta) {
            throw std::runtime_error("Failed to load root segment: " + segment_id);
        }

        QuadSurface* root_surf = root_meta->surface();

        // Get or generate mask and image for root
        auto [root_image, root_mask] = loadOrGenerateMaskedImage(root_surf, root_meta->path);

        // Get overlapping segments based on source
        std::vector<std::string> overlap_ids = getOverlapIds(segment_id, root_meta, source);

        std::cout << "Found " << overlap_ids.size() << " overlapping segments from source: " << source << std::endl;
        std::vector<SegmentData> overlaps;

        // Process each overlap
        int color_idx = 0;
        int num_processed = 0;
        for (const std::string& overlap_id : overlap_ids) {
            num_processed++;
            if (num_processed > 5) break;
            std::cout << "Processing overlap: " << overlap_id << std::endl;

            auto overlap_meta = vpkg_->loadSurface(overlap_id);
            if (!overlap_meta) {
                std::cerr << "Failed to load overlap: " << overlap_id << std::endl;
                continue;
            }

            QuadSurface* overlap_surf = overlap_meta->surface();

            // Find alignment
            AlignmentResult alignment = findAlignment(root_surf, overlap_surf);

            if (!alignment.valid) {
                std::cerr << "Failed to align: " << overlap_id << std::endl;
                continue;
            }

            std::cout << "Alignment found with " << alignment.num_correspondences
                     << " points, offset: " << alignment.pixel_offset << std::endl;

            // Get or generate mask and image for overlap
            auto [overlap_image, overlap_mask] = loadOrGenerateMaskedImage(overlap_surf, overlap_meta->path);

            overlaps.push_back({
                overlap_image,
                overlap_mask,
                alignment.pixel_offset,
                tint_colors_[color_idx % tint_colors_.size()]
            });
            color_idx++;
        }

        // Composite all segments
        cv::Mat final_image = compositeSegments(root_image, root_mask, overlaps, opacity);

        // Save output
        cv::imwrite(output_path.string(), final_image);
        std::cout << "Saved to: " << output_path << std::endl;

        return final_image;
    }

private:
    cv::Mat renderSequence(const std::string& target_segment_id, const fs::path& output_path, float opacity) {
        std::cout << "Rendering sequence with target segment: " << target_segment_id << std::endl;

        // Load the target segment's metadata to get seed and sequence
        auto target_meta = vpkg_->loadSurface(target_segment_id);
        if (!target_meta) {
            throw std::runtime_error("Failed to load target segment: " + target_segment_id);
        }

        // Load meta.json to get seed and surface_sequence
        fs::path meta_path = target_meta->path / "meta.json";
        if (!fs::exists(meta_path)) {
            throw std::runtime_error("meta.json not found for segment: " + target_segment_id);
        }

        std::ifstream meta_file(meta_path);
        json meta_json;
        meta_file >> meta_json;

        if (!meta_json.contains("seed")) {
            throw std::runtime_error("seed not found in meta.json for segment: " + target_segment_id);
        }
        if (!meta_json.contains("surface_sequence")) {
            throw std::runtime_error("surface_sequence not found in meta.json for segment: " + target_segment_id);
        }

        std::string seed_id = meta_json["seed"].get<std::string>();
        auto sequence = meta_json["surface_sequence"].get<std::vector<std::string>>();

        std::cout << "Using seed as root: " << seed_id << std::endl;
        std::cout << "Processing sequence of " << sequence.size() << " segments" << std::endl;
        std::cout << "Using write-once mode to prevent overlapping pixels" << std::endl;

        // Load seed segment as the root (at origin)
        auto seed_meta = vpkg_->loadSurface(seed_id);
        if (!seed_meta) {
            throw std::runtime_error("Failed to load seed segment: " + seed_id);
        }

        QuadSurface* seed_surf = seed_meta->surface();
        auto [root_image, root_mask] = loadOrGenerateMaskedImage(seed_surf, seed_meta->path);

        // Prepare overlaps vector
        std::vector<SegmentData> overlaps;
        int color_idx = 0;

        // Process sequence segments until we reach the target
        for (const std::string& seq_id : sequence) {
            std::cout << "Processing sequence segment: " << seq_id << std::endl;

            auto seq_meta = vpkg_->loadSurface(seq_id);
            if (!seq_meta) {
                std::cerr << "Failed to load sequence segment: " << seq_id << std::endl;
                continue;
            }

            QuadSurface* seq_surf = seq_meta->surface();

            // Find alignment relative to seed
            AlignmentResult alignment = findAlignment(seed_surf, seq_surf);

            if (!alignment.valid) {
                std::cerr << "Failed to align sequence segment: " << seq_id << std::endl;
                continue;
            }

            std::cout << "Alignment found with " << alignment.num_correspondences
                     << " points, offset: " << alignment.pixel_offset << std::endl;

            // Get or generate mask and image
            auto [seq_image, seq_mask] = loadOrGenerateMaskedImage(seq_surf, seq_meta->path);

            overlaps.push_back({
                seq_image,
                seq_mask,
                alignment.pixel_offset,
                tint_colors_[color_idx % tint_colors_.size()]
            });
            color_idx++;

            // Stop if this segment matches the target segment
            if (seq_id == target_segment_id) {
                std::cout << "Reached target segment, stopping sequence" << std::endl;
                break;
            }
        }

        // If target segment wasn't in the sequence, add it as the final overlay
        bool found_target = false;
        for (const auto& seq_id : sequence) {
            if (seq_id == target_segment_id) {
                found_target = true;
                break;
            }
        }

        if (!found_target) {
            std::cout << "Target segment not in sequence, adding as final overlay" << std::endl;

            QuadSurface* target_surf = target_meta->surface();
            AlignmentResult alignment = findAlignment(seed_surf, target_surf);

            if (alignment.valid) {
                auto [target_image, target_mask] = loadOrGenerateMaskedImage(target_surf, target_meta->path);
                overlaps.push_back({
                    target_image,
                    target_mask,
                    alignment.pixel_offset,
                    tint_colors_[color_idx % tint_colors_.size()]
                });
            } else {
                std::cerr << "Failed to align target segment: " << target_segment_id << std::endl;
            }
        }

        // Composite all segments with seed as base
        cv::Mat final_image = compositeSegments(root_image, root_mask, overlaps, opacity);

        // Save output
        cv::imwrite(output_path.string(), final_image);
        std::cout << "Saved to: " << output_path << std::endl;

        return final_image;
    }

    std::vector<std::string> getOverlapIds(const std::string& root_id,
                                          std::shared_ptr<SurfaceMeta> root_meta,
                                          std::string source) {
        std::vector<std::string> overlap_ids;

        if (source == "overlapping") {
            // Original behavior - use overlapping.json
            root_meta->readOverlapping();
            if (root_meta->overlapping_str.empty()) {
                throw std::runtime_error("No overlapping segments found in overlapping.json for segment: " + root_id);
            }
            //overlap_ids = root_meta->overlapping_str;
        } else if (source == "contributing")  {
            // Load from meta.json contributing_surfaces field
            fs::path meta_path = root_meta->path / "meta.json";
            if (!fs::exists(meta_path)) {
                throw std::runtime_error("meta.json not found for segment: " + root_id);
            }

            std::ifstream meta_file(meta_path);
            json meta_json;
            meta_file >> meta_json;

            if (!meta_json.contains("contributing_surfaces")) {
                throw std::runtime_error("contributing_surfaces not found in meta.json for segment: " + root_id);
            }

            overlap_ids = meta_json["contributing_surfaces"].get<std::vector<std::string>>();
            if (overlap_ids.empty()) {
                throw std::runtime_error("contributing_surfaces is empty in meta.json for segment: " + root_id);
            }
        }
        // Note: "sequence" is now handled separately in render() method

        return overlap_ids;
    }

    std::pair<cv::Mat_<cv::Vec3b>, cv::Mat_<uint8_t>> loadOrGenerateMaskedImage(
        QuadSurface* surf, const fs::path& segment_path) {

        cv::Mat_<uint8_t> mask;
        cv::Mat_<uint8_t> img;
        fs::path mask_path = segment_path / "mask.tif";

        // Check if mask.tif exists
        //TODO: fixme
        /*
        if (fs::exists(mask_path)) {
            std::cout << "Loading existing mask from: " << mask_path << std::endl;
            std::vector<cv::Mat> layers;
            cv::imreadmulti(mask_path.string(), layers, cv::IMREAD_GRAYSCALE);
            if (layers.size() == 2) {
                mask = layers[1];
                cv::Size surf_size = surf->size();
                if (mask.size() != surf_size) {
                    cv::resize(mask, mask, surf_size, 0, 0, cv::INTER_NEAREST);
                }
            }
        }*/

        // If no mask loaded, generate it along with the image
        if (mask.empty()) {
            std::cout << "Generating mask and image data" << std::endl;

            // Use the actual volume's datasets
            z5::Dataset* ds_high = volume_->zarrDataset(0);  // Full resolution
            z5::Dataset* ds_low = nullptr;
            if (volume_->numScales() > 2) {
                ds_low = volume_->zarrDataset(2);  // Lower resolution for large surfaces
            }

            generate_mask(surf, mask, img, ds_high, ds_low, cache_);

            // Optionally save the generated mask for future use
            if (!mask_path.parent_path().empty()) {
                cv::imwrite(mask_path.string(), mask);
                std::cout << "Saved generated mask to: " << mask_path << std::endl;
            }
        } else {
            // We have a mask but need to generate the image data
            std::cout << "Generating image data for existing mask" << std::endl;

            cv::Size native_size = surf->size();
            cv::Mat_<cv::Vec3f> coords;
            cv::Vec3f center = surf->pointer();
            surf->gen(&coords, nullptr, native_size, center, 1.0f, {0, 0, 0});

            // Choose appropriate scale based on surface size
            int ds_idx = 0;
            if (native_size.width >= 4000 && volume_->numScales() > 2) {
                ds_idx = 2;  // Use lower resolution for large surfaces
            } else if (native_size.width >= 2000 && volume_->numScales() > 1) {
                ds_idx = 1;  // Medium resolution
            }

            float ds_scale = std::pow(2.0f, -ds_idx);
            readInterpolated3D(img, volume_->zarrDataset(ds_idx), coords * ds_scale, cache_);

            // If we downsampled, resize back
            if (ds_idx > 0) {
                cv::resize(img, img, native_size, 0, 0, cv::INTER_LINEAR);
            }
        }

        // Convert to RGB
        cv::Mat_<cv::Vec3b> rgb_image;
        cv::cvtColor(img, rgb_image, cv::COLOR_GRAY2RGB);

        // Apply mask to ensure invalid regions are black
        for (int y = 0; y < rgb_image.rows; y++) {
            for (int x = 0; x < rgb_image.cols; x++) {
                if (!mask(y, x)) {
                    rgb_image(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        return {rgb_image, mask};
    }

    AlignmentResult findAlignment(QuadSurface* ref_surf, QuadSurface* target_surf) {
        AlignmentResult result;
        result.valid = false;
        result.num_correspondences = 0;

        // Get the intersection
        QuadSurface* intersection = surface_intersection(ref_surf, target_surf, 2.0);
        if (!intersection) {
            return result;
        }

        cv::Mat_<cv::Vec3f> intersect_points = intersection->rawPoints();
        std::vector<cv::Vec2f> ref_coords;
        std::vector<cv::Vec2f> target_coords;

        // Sample points from intersection
        int step = std::max(10, std::min(intersect_points.rows, intersect_points.cols) / 20);

        for (int j = step; j < intersect_points.rows - step; j += step) {
            for (int i = step; i < intersect_points.cols - step; i += step) {
                cv::Vec3f point = intersect_points(j, i);
                if (point[0] == -1) continue;

                cv::Vec3f ref_ptr = ref_surf->pointer();
                cv::Vec3f target_ptr = target_surf->pointer();

                float ref_dist = ref_surf->pointTo(ref_ptr, point, 2.0, 1000);
                float target_dist = target_surf->pointTo(target_ptr, point, 2.0, 1000);

                if (ref_dist >= 0 && ref_dist <= 2.0 && target_dist >= 0 && target_dist <= 2.0) {
                    cv::Vec3f ref_loc = ref_surf->loc_raw(ref_ptr);
                    cv::Vec3f target_loc = target_surf->loc_raw(target_ptr);

                    ref_coords.push_back(cv::Vec2f(ref_loc[0], ref_loc[1]));
                    target_coords.push_back(cv::Vec2f(target_loc[0], target_loc[1]));
                }

                if (ref_coords.size() >= 50) break;
            }
            if (ref_coords.size() >= 50) break;
        }

        delete intersection;

        if (ref_coords.size() < 3) {
            return result;
        }

        // Calculate robust offset
        std::vector<float> x_offsets, y_offsets;
        for (size_t k = 0; k < ref_coords.size(); k++) {
            x_offsets.push_back(ref_coords[k][0] - target_coords[k][0]);
            y_offsets.push_back(ref_coords[k][1] - target_coords[k][1]);
        }

        std::sort(x_offsets.begin(), x_offsets.end());
        std::sort(y_offsets.begin(), y_offsets.end());

        result.pixel_offset = cv::Vec2f(x_offsets[x_offsets.size()/2],
                                       y_offsets[y_offsets.size()/2]);
        result.num_correspondences = ref_coords.size();
        result.valid = true;

        return result;
    }

    cv::Mat compositeSegments(const cv::Mat_<cv::Vec3b>& root_image,
                             const cv::Mat_<uint8_t>& root_mask,
                             const std::vector<SegmentData>& overlaps,
                             float opacity) {

        // Calculate bounding box
        int min_x = 0, min_y = 0;
        int max_x = root_image.cols, max_y = root_image.rows;

        for (const auto& overlap : overlaps) {
            min_x = std::min(min_x, (int)overlap.offset[0]);
            min_y = std::min(min_y, (int)overlap.offset[1]);
            max_x = std::max(max_x, (int)(overlap.offset[0] + overlap.image.cols));
            max_y = std::max(max_y, (int)(overlap.offset[1] + overlap.image.rows));
        }

        // Add padding
        int padding = 50;
        cv::Size output_size(max_x - min_x + 2*padding, max_y - min_y + 2*padding);
        cv::Mat_<cv::Vec3b> output(output_size, cv::Vec3b(0, 0, 0));

        cv::Vec2f global_offset(-min_x + padding, -min_y + padding);

        // Draw root
        for (int y = 0; y < root_image.rows; y++) {
            for (int x = 0; x < root_image.cols; x++) {
                if (root_mask(y, x)) {
                    int out_x = x + global_offset[0];
                    int out_y = y + global_offset[1];
                    output(out_y, out_x) = root_image(y, x);
                }
            }
        }

        // Overlay the overlaps with tinting and transparency
        for (const auto& overlap : overlaps) {
            for (int y = 0; y < overlap.image.rows; y++) {
                for (int x = 0; x < overlap.image.cols; x++) {
                    if (overlap.mask(y, x)) {
                        int out_x = x + overlap.offset[0] + global_offset[0];
                        int out_y = y + overlap.offset[1] + global_offset[1];

                        if (out_x >= 0 && out_x < output.cols &&
                            out_y >= 0 && out_y < output.rows) {
                            cv::Vec3b& dst = output(out_y, out_x);
                            cv::Vec3b src = overlap.image(y, x);

                            // Apply tint to the overlap
                            for (int c = 0; c < 3; c++) {
                                src[c] = std::min(255, src[c] + overlap.tint[c]);
                            }

                            // Blend with transparency
                            for (int c = 0; c < 3; c++) {
                                dst[c] = dst[c] * (1 - opacity) + src[c] * opacity;
                            }
                        }
                    }
                }
            }
        }

        return output;
    }
};


int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <volpkg-path> <volume-id> <segment-id> <overlap-source> <output-png> " << std::endl;
        std::cout << "  volpkg-path: Path to volume package" << std::endl;
        std::cout << "  volume-id: ID of volume to use" << std::endl;
        std::cout << "  segment-id: ID of segment to render" << std::endl;
        std::cout << "  overlap-source: Source for overlaps (overlapping|contributing|sequence)" << std::endl;
        std::cout << "  output-png: Output file path" << std::endl;
        return EXIT_SUCCESS;
    }

    fs::path volpkg_path = argv[1];
    std::string volume_id = argv[2];
    std::string segment_id = argv[3];
    std::string overlap_source = argv[4];
    fs::path output_path = argv[5];
    float opacity = 0.8f;

    // Parse overlap source
    if (overlap_source != "overlapping" && overlap_source != "sequence" && overlap_source != "contributing") {
        std::cerr << "Error: Invalid overlap source. Must be one of: overlapping, contributing, sequence" << std::endl;
        return EXIT_FAILURE;
    }

    // Validate opacity
    if (opacity < 0.0f || opacity > 1.0f) {
        std::cerr << "Error: Opacity must be between 0.0 and 1.0" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        SegmentRenderer renderer(volpkg_path, volume_id);
        renderer.render(segment_id, output_path, overlap_source, opacity);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}