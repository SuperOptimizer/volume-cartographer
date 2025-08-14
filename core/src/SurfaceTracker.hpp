#pragma once

#include <opencv2/core.hpp>
#include <ceres/ceres.h>
#include <unordered_map>
#include <set>
#include <shared_mutex>
#include <nlohmann/json.hpp>
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"

// Forward declarations
class SurfaceMeta;
class QuadSurface;
template<typename T, typename C> class Chunked3d;

class SurfaceTracker {
public:
    // Inner types
    using SurfPoint = std::pair<SurfaceMeta*, cv::Vec2i>;

    class resId_t {
    public:
        resId_t();
        resId_t(int type, SurfaceMeta* sm, cv::Vec2i p);
        resId_t(int type, SurfaceMeta* sm, const cv::Vec2i& a, const cv::Vec2i& b);
        bool operator==(const resId_t& o) const;

        int _type;
        SurfaceMeta* _sm;
        cv::Vec2i _p;
    };

    struct resId_hash {
        size_t operator()(resId_t id) const;
    };

    struct SurfPoint_hash {
        size_t operator()(SurfPoint p) const;
    };

    struct vec2i_hash {
        size_t operator()(cv::Vec2i p) const;
    };

    // Constructor/Destructor
    SurfaceTracker();
    SurfaceTracker(const nlohmann::json& params);
    ~SurfaceTracker();
    SurfaceTracker(const SurfaceTracker& other)
    : _data(other._data),
      _res_blocks(other._res_blocks),
      _surfs(other._surfs),
      _emptysurfs(other._emptysurfs),
      seed_coord(other.seed_coord),
      seed_loc(other.seed_loc),
      local_cost_inl_th(other.local_cost_inl_th),
      same_surface_th(other.same_surface_th),
      straight_weight(other.straight_weight),
      straight_weight_3D(other.straight_weight_3D),
      sliding_w_scale(other.sliding_w_scale),
      z_loc_loss_w(other.z_loc_loss_w),
      dist_loss_2d_w(other.dist_loss_2d_w),
      dist_loss_3d_w(other.dist_loss_3d_w),
      straight_min_count(other.straight_min_count),
      inlier_base_threshold(other.inlier_base_threshold)
    // mutex is default-constructed (not copied)
{}

    SurfaceTracker& operator=(const SurfaceTracker& other) {
        if (this != &other) {
            // Don't copy mutex, just copy data
            _data = other._data;
            _res_blocks = other._res_blocks;
            _surfs = other._surfs;
            _emptysurfs = other._emptysurfs;
            seed_coord = other.seed_coord;
            seed_loc = other.seed_loc;
            local_cost_inl_th = other.local_cost_inl_th;
            same_surface_th = other.same_surface_th;
            straight_weight = other.straight_weight;
            straight_weight_3D = other.straight_weight_3D;
            sliding_w_scale = other.sliding_w_scale;
            z_loc_loss_w = other.z_loc_loss_w;
            dist_loss_2d_w = other.dist_loss_2d_w;
            dist_loss_3d_w = other.dist_loss_3d_w;
            straight_min_count = other.straight_min_count;
            inlier_base_threshold = other.inlier_base_threshold;
        }
        return *this;
    }

    // Data access methods
    cv::Vec2d& loc(SurfaceMeta* sm, const cv::Vec2i& loc);
    ceres::ResidualBlockId& resId(const resId_t& id);
    bool hasResId(const resId_t& id);
    bool has(SurfaceMeta* sm, const cv::Vec2i& loc) const;
    void erase(SurfaceMeta* sm, const cv::Vec2i& loc);
    void eraseSurf(SurfaceMeta* sm, const cv::Vec2i& loc);
    std::set<SurfaceMeta*>& surfs(const cv::Vec2i& loc);
    const std::set<SurfaceMeta*>& surfsC(const cv::Vec2i& loc) const;

    // Lookup methods
    cv::Vec3d lookupInt(SurfaceMeta* sm, const cv::Vec2i& p);
    bool validInt(SurfaceMeta* sm, const cv::Vec2i& p);
    cv::Vec3d lookupIntLoc(SurfaceMeta* sm, const cv::Vec2f& l);

    // Transformation methods
    void flipX(int x0);
    void copy(const SurfaceTracker& src, const cv::Rect& roi);

    // Loss functions - 2D
    int addDistLoss(SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& off,
                    ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                    float unit, int flags = 0, ceres::ResidualBlockId* res = nullptr, float w = 1.0f);

    int addStraightLoss(SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& o1,
                        const cv::Vec2i& o2, const cv::Vec2i& o3,
                        ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                        int flags = 0, float w = 0.7f);

    int addSurfLoss(SurfaceMeta* sm, const cv::Vec2i p,
                    ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                    cv::Mat_<cv::Vec3d>& points, float step,
                    ceres::ResidualBlockId* res = nullptr, float w = 0.1f);

    // Loss functions - 3D
    int addDistLoss3D(cv::Mat_<cv::Vec3d>& points, const cv::Vec2i& p, const cv::Vec2i& off,
                      ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                      float unit, int flags = 0, ceres::ResidualBlockId* res = nullptr, float w = 2.0f);

    int addStraightLoss3D(const cv::Vec2i& p, const cv::Vec2i& o1, const cv::Vec2i& o2,
                          const cv::Vec2i& o3, cv::Mat_<cv::Vec3d>& points,
                          ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                          int flags = 0, ceres::ResidualBlockId* res = nullptr, float w = 4.0f);

    // Conditional loss functions
    int condDistLoss(int type, SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& off,
                     ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                     float unit, int flags = 0);

    int condDistLoss3D(int type, SurfaceMeta* sm, cv::Mat_<cv::Vec3d>& points,
                       const cv::Vec2i& p, const cv::Vec2i& off,
                       ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                       float unit, int flags = 0);

    int condStraightLoss3D(int type, SurfaceMeta* sm, const cv::Vec2i& p,
                           const cv::Vec2i& o1, const cv::Vec2i& o2, const cv::Vec2i& o3,
                           cv::Mat_<cv::Vec3d>& points, ceres::Problem& problem,
                           const cv::Mat_<uint8_t>& state, int flags = 0);

    int condSurfLoss(int type, SurfaceMeta* sm, const cv::Vec2i p,
                     ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                     cv::Mat_<cv::Vec3d>& points, float step);

    // Space loss functions (template methods)
    template <typename T, typename C>
    int addSpaceLoss(ceres::Problem& problem, const cv::Vec2i& p,
                     cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                     Chunked3d<T,C>& t, float w = 0.1f);

    template <typename T, typename C>
    int addSpaceLineLoss(ceres::Problem& problem, const cv::Vec2i& p, const cv::Vec2i& off,
                         cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                         Chunked3d<T,C>& t, int steps, float w = 0.1f, float dist_th = 2);

    template <typename T, typename C>
    int condSpaceLineLoss(int bit, const cv::Vec2i& p, const cv::Vec2i& off,
                          cv::Mat_<uint16_t>& loss_status, ceres::Problem& problem,
                          cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                          Chunked3d<T,C>& t, int steps);

    // Cost and optimization functions
    double localCost(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                     cv::Mat_<cv::Vec3d>& points, float step, float src_step,
                     int* ref_count = nullptr, int* straight_count_ptr = nullptr);

    double localCostDestructive(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                                cv::Mat_<cv::Vec3d>& points, float step, float src_step,
                                cv::Vec3f loc, int* ref_count = nullptr, int* straight_count_ptr = nullptr);

    double localSolve(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                      cv::Mat_<cv::Vec3d>& points, float step, float src_step, int flags);

    // Add local/global functions
    int addLocal(SurfaceMeta* sm, const cv::Vec2i p, ceres::Problem& problem,
                 const cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                 float step, float src_step, int flags = 0, int* straight_count_ptr = nullptr);

    int addGlobal(SurfaceMeta* sm, const cv::Vec2i p, ceres::Problem& problem,
                  const cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                  float step, int flags = 0, float step_onsurf = 0);

    // Point generation
    cv::Mat_<cv::Vec3d> genPointsHR(cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                                     cv::Rect& used_area, float step, float step_src,
                                     bool inpaint = false);

    // Surface optimization
    void optimizeSurfaceMapping(cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                                cv::Rect used_area, cv::Rect static_bounds,
                                float step, float src_step, const cv::Vec2i& seed,
                                int closing_r, bool keep_inpainted = false,
                                const std::filesystem::path& tgt_dir = std::filesystem::path());

    // Thread safety
    SurfaceTracker* getThreadCopy();
    void mergeThreadData(const SurfaceTracker& thread_tracker, const std::vector<cv::Vec2i>& points);

    // Seed management
    void setSeed(const cv::Vec3d& coord, const cv::Vec2i& loc);
    cv::Vec3d getSeedCoord() const { return seed_coord; }
    cv::Vec2i getSeedLoc() const { return seed_loc; }

    // Configuration access
    float getLocalCostInlTh() const { return local_cost_inl_th; }
    float getSameSurfaceTh() const { return same_surface_th; }

private:
    // Data members
    std::unordered_map<SurfPoint, cv::Vec2d, SurfPoint_hash> _data;
    std::unordered_map<resId_t, ceres::ResidualBlockId, resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i, std::set<SurfaceMeta*>, vec2i_hash> _surfs;
    std::set<SurfaceMeta*> _emptysurfs;
    cv::Vec3d seed_coord;
    cv::Vec2i seed_loc;

    // Configuration parameters
    float local_cost_inl_th;
    float same_surface_th;
    float straight_weight;
    float straight_weight_3D;
    float sliding_w_scale;
    float z_loc_loss_w;
    float dist_loss_2d_w;
    float dist_loss_3d_w;
    float straight_min_count;
    int inlier_base_threshold;

    // Thread safety
    mutable std::shared_mutex mutex;

    // Helper methods
    bool lossMask(int bit, const cv::Vec2i& p, const cv::Vec2i& off,
                  cv::Mat_<uint16_t>& loss_status) const;
    int setLossMask(int bit, const cv::Vec2i& p, const cv::Vec2i& off,
                    cv::Mat_<uint16_t>& loss_status, int set);
    cv::Vec2i lowerP(const cv::Vec2i& point, const cv::Vec2i& offset) const;
};

// Template method implementations (in header file)
template <typename T, typename C>
int SurfaceTracker::addSpaceLoss(ceres::Problem& problem, const cv::Vec2i& p,
                                  cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                                  Chunked3d<T,C>& t, float w)
{
    if (!(state(p) & STATE_LOC_VALID))
        return 0;

    problem.AddResidualBlock(SpaceLossAcc<T,C>::Create(t, w), nullptr, &loc(p)[0]);

    return 1;
}

template <typename T, typename C>
int SurfaceTracker::addSpaceLineLoss(ceres::Problem& problem, const cv::Vec2i& p, const cv::Vec2i& off,
                                      cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                                      Chunked3d<T,C>& t, int steps, float w, float dist_th)
{
    if (!(state(p) & STATE_LOC_VALID))
        return 0;
    if (!(state(p+off) & STATE_LOC_VALID))
        return 0;

    problem.AddResidualBlock(SpaceLineLossAcc<T,C>::Create(t, steps, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

template <typename T, typename C>
int SurfaceTracker::condSpaceLineLoss(int bit, const cv::Vec2i& p, const cv::Vec2i& off,
                                       cv::Mat_<uint16_t>& loss_status, ceres::Problem& problem,
                                       cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& loc,
                                       Chunked3d<T,C>& t, int steps)
{
    int set = 0;
    if (!lossMask(bit, p, off, loss_status))
        set = setLossMask(bit, p, off, loss_status, addSpaceLineLoss(problem, p, off, state, loc, t, steps));
    return set;
}

