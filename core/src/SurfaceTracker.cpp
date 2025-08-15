#include "SurfaceTracker.hpp"

#include <omp.h>
#include <opencv2/imgproc.hpp>

#include "vc/core/math/MathUtils.hpp"
#include "vc/core/util/Lifetime.hpp"
#include "vc/core/util/DateTime.hpp"

// Constructor/Destructor
SurfaceTracker::SurfaceTracker()
    : local_cost_inl_th(0.2f),
      same_surface_th(2.0f),
      straight_weight(0.7f),
      straight_weight_3D(4.0f),
      sliding_w_scale(1.0f),
      z_loc_loss_w(0.1f),
      dist_loss_2d_w(1.0f),
      dist_loss_3d_w(2.0f),
      straight_min_count(1.0f),
      inlier_base_threshold(20),
      seed_coord(-1, -1, -1),
      seed_loc(-1, -1)
{
}

SurfaceTracker::SurfaceTracker(const nlohmann::json& params)
    : local_cost_inl_th(params.value("local_cost_inl_th", 0.2f)),
      same_surface_th(params.value("same_surface_th", 2.0f)),
      straight_weight(params.value("straight_weight", 0.7f)),
      straight_weight_3D(params.value("straight_weight_3D", 4.0f)),
      sliding_w_scale(params.value("sliding_w_scale", 1.0f)),
      z_loc_loss_w(params.value("z_loc_loss_w", 0.1f)),
      dist_loss_2d_w(params.value("dist_loss_2d_w", 1.0f)),
      dist_loss_3d_w(params.value("dist_loss_3d_w", 2.0f)),
      straight_min_count(params.value("straight_min_count", 1.0f)),
      inlier_base_threshold(params.value("inlier_base_threshold", 20)),
      seed_coord(-1, -1, -1),
      seed_loc(-1, -1)
{
}

// resId_t implementations
SurfaceTracker::resId_t::resId_t() : _type(0), _sm(nullptr), _p(-1, -1) {}

SurfaceTracker::resId_t::resId_t(int type, SurfaceMeta* sm, cv::Vec2i p)
    : _type(type), _sm(sm), _p(p) {}

SurfaceTracker::resId_t::resId_t(int type, SurfaceMeta* sm, const cv::Vec2i& a, const cv::Vec2i& b)
    : _type(type), _sm(sm), _p((a[0]+b[0])*1000 + a[1]+b[1]) {}

bool SurfaceTracker::resId_t::operator==(const resId_t& o) const {
    return _type == o._type && _sm == o._sm && _p[0] == o._p[0] && _p[1] == o._p[1];
}

// Hash function implementations
size_t SurfaceTracker::resId_hash::operator()(resId_t id) const {
    size_t hash1 = std::hash<int>{}(id._type);
    size_t hash2 = std::hash<void*>{}(id._sm);
    size_t hash3 = std::hash<int>{}(id._p[0]);
    size_t hash4 = std::hash<int>{}(id._p[1]);

    size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    hash = hash ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

    return hash;
}

size_t SurfaceTracker::SurfPoint_hash::operator()(SurfPoint p) const {
    size_t hash1 = std::hash<void*>{}(p.first);
    size_t hash2 = std::hash<int>{}(p.second[0]);
    size_t hash3 = std::hash<int>{}(p.second[1]);

    size_t hash = hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    hash = hash ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

    return hash;
}

size_t SurfaceTracker::vec2i_hash::operator()(cv::Vec2i p) const {
    size_t hash1 = std::hash<int>{}(p[0]);
    size_t hash2 = std::hash<int>{}(p[1]);
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
}

SurfaceTracker::~SurfaceTracker()
{
}

// Data access methods
cv::Vec2d& SurfaceTracker::loc(SurfaceMeta* sm, const cv::Vec2i& loc)
{
    return _data[{sm, loc}];
}

ceres::ResidualBlockId& SurfaceTracker::resId(const resId_t& id)
{
    return _res_blocks[id];
}

bool SurfaceTracker::hasResId(const resId_t& id)
{
    return _res_blocks.count(id);
}

bool SurfaceTracker::has(SurfaceMeta* sm, const cv::Vec2i& loc) const
{
    return _data.count({sm, loc});
}

void SurfaceTracker::erase(SurfaceMeta* sm, const cv::Vec2i& loc)
{
    _data.erase({sm, loc});
}

void SurfaceTracker::eraseSurf(SurfaceMeta* sm, const cv::Vec2i& loc)
{
    _surfs[loc].erase(sm);
}

std::set<SurfaceMeta*>& SurfaceTracker::surfs(const cv::Vec2i& loc)
{
    return _surfs[loc];
}

const std::set<SurfaceMeta*>& SurfaceTracker::surfsC(const cv::Vec2i& loc) const
{
    if (!_surfs.count(loc))
        return _emptysurfs;
    else
        return _surfs.find(loc)->second;
}

// Lookup methods
cv::Vec3d SurfaceTracker::lookupInt(SurfaceMeta* sm, const cv::Vec2i& p)
{
    auto id = std::make_pair(sm, p);
    if (!_data.count(id))
        throw std::runtime_error("error, lookup failed!");
    cv::Vec2d l = loc(sm, p);
    if (l[0] == -1)
        return {-1, -1, -1};
    else {
        cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2, sm->surface()->rawPoints().cols-2};
        cv::Vec2i li = {floor(l[0]), floor(l[1])};
        if (bounds.contains(cv::Point(li)))
            return at_int_inv(sm->surface()->rawPoints(), l);
        else
            return {-1, -1, -1};
    }
}

bool SurfaceTracker::validInt(SurfaceMeta* sm, const cv::Vec2i& p)
{
    auto id = std::make_pair(sm, p);
    if (!_data.count(id))
        return false;
    cv::Vec2d l = loc(sm, p);
    if (l[0] == -1)
        return false;
    else {
        cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2, sm->surface()->rawPoints().cols-2};
        cv::Vec2i li = {floor(l[0]), floor(l[1])};
        if (bounds.contains(cv::Point(li)))
        {
            if (sm->surface()->rawPoints()(li[0], li[1])[0] == -1)
                return false;
            if (sm->surface()->rawPoints()(li[0]+1, li[1])[0] == -1)
                return false;
            if (sm->surface()->rawPoints()(li[0], li[1]+1)[0] == -1)
                return false;
            if (sm->surface()->rawPoints()(li[0]+1, li[1]+1)[0] == -1)
                return false;
            return true;
        }
        else
            return false;
    }
}

cv::Vec3d SurfaceTracker::lookupIntLoc(SurfaceMeta* sm, const cv::Vec2f& l)
{
    if (l[0] == -1)
        return {-1, -1, -1};
    else {
        cv::Rect bounds = {0, 0, sm->surface()->rawPoints().rows-2, sm->surface()->rawPoints().cols-2};
        if (bounds.contains(cv::Point(l)))
            return at_int_inv(sm->surface()->rawPoints(), l);
        else
            return {-1, -1, -1};
    }
}

// Transformation methods
void SurfaceTracker::flipX(int x0)
{
    std::cout << " src sizes " << _data.size() << " " << _surfs.size() << std::endl;
    SurfaceTracker old = *this;
    _data.clear();
    _res_blocks.clear();
    _surfs.clear();

    for(auto& it : old._data)
        _data[{it.first.first, {it.first.second[0], x0+x0-it.first.second[1]}}] = it.second;

    for(auto& it : old._surfs)
        _surfs[{it.first[0], x0+x0-it.first[1]}] = it.second;

    std::cout << " flipped sizes " << _data.size() << " " << _surfs.size() << std::endl;
}

void SurfaceTracker::copy(const SurfaceTracker& src, const cv::Rect& roi_)
{
    cv::Rect roi(roi_.y, roi_.x, roi_.height, roi_.width);

    {
        auto it = _data.begin();
        while (it != _data.end()) {
            if (roi.contains(cv::Point(it->first.second)))
                it = _data.erase(it);
            else
                it++;
        }
    }

    {
        auto it = _surfs.begin();
        while (it != _surfs.end()) {
            if (roi.contains(cv::Point(it->first)))
                it = _surfs.erase(it);
            else
                it++;
        }
    }

    for(auto& it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            _data[it.first] = it.second;
    for(auto& it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            _surfs[it.first] = it.second;
}

// Loss functions - 2D
int SurfaceTracker::addDistLoss(SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& off,
                                ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                float unit, int flags, ceres::ResidualBlockId* res, float w)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !has(sm, p))
        return 0;
    if ((state(p+off) & STATE_LOC_VALID) == 0 || !has(sm, p+off))
        return 0;

    // Use the member variable if w is default value (1.0), otherwise use the provided value
    float weight = (w == 1.0f) ? dist_loss_2d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        DistLoss2D::Create(unit*cv::norm(off), weight), nullptr,
        &loc(sm, p)[0], &loc(sm, p+off)[0]);

    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&loc(sm, p+off)[0]);

    return 1;
}

int SurfaceTracker::addStraightLoss(SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& o1,
                                    const cv::Vec2i& o2, const cv::Vec2i& o3,
                                    ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                    int flags, float w)
{
    if ((state(p+o1) & STATE_LOC_VALID) == 0 || !has(sm, p+o1))
        return 0;
    if ((state(p+o2) & STATE_LOC_VALID) == 0 || !has(sm, p+o2))
        return 0;
    if ((state(p+o3) & STATE_LOC_VALID) == 0 || !has(sm, p+o3))
        return 0;

    // Always use the member variable straight_weight for 2D
    w = straight_weight;

    problem.AddResidualBlock(StraightLoss2D::Create(w), nullptr,
        &loc(sm, p+o1)[0], &loc(sm, p+o2)[0], &loc(sm, p+o3)[0]);

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&loc(sm, p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&loc(sm, p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&loc(sm, p+o3)[0]);
    }

    return 1;
}

int SurfaceTracker::addSurfLoss(SurfaceMeta* sm, const cv::Vec2i p,
                                ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                cv::Mat_<cv::Vec3d>& points, float step,
                                ceres::ResidualBlockId* res, float w)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !validInt(sm, p))
        return 0;

    ceres::ResidualBlockId tmp;
    tmp = problem.AddResidualBlock(
        SurfaceLossD::Create(sm->surface()->rawPoints(), w), nullptr,
        &points(p)[0], &loc(sm, p)[0]);

    if (res)
        *res = tmp;

    return 1;
}

// Loss functions - 3D
int SurfaceTracker::addDistLoss3D(cv::Mat_<cv::Vec3d>& points, const cv::Vec2i& p, const cv::Vec2i& off,
                                   ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                   float unit, int flags, ceres::ResidualBlockId* res, float w)
{
    if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0)
        return 0;

    // Use the member variable if w is default value (2.0), otherwise use the provided value
    float weight = (w == 2.0f) ? dist_loss_3d_w : w;
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        DistLoss::Create(unit*cv::norm(off), weight), nullptr,
        &points(p)[0], &points(p+off)[0]);

    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(p+off)[0]);

    return 1;
}

int SurfaceTracker::addStraightLoss3D(const cv::Vec2i& p, const cv::Vec2i& o1, const cv::Vec2i& o2,
                                       const cv::Vec2i& o3, cv::Mat_<cv::Vec3d>& points,
                                       ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                       int flags, ceres::ResidualBlockId* res, float w)
{
    if ((state(p+o1) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o2) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o3) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;

    // Always use the member variable straight_weight_3D for 3D
    w = straight_weight_3D;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(
        StraightLoss::Create(w), nullptr,
        &points(p+o1)[0], &points(p+o2)[0], &points(p+o3)[0]);

    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o3)[0]);
    }

    return 1;
}

// Conditional loss functions
int SurfaceTracker::condDistLoss(int type, SurfaceMeta* sm, const cv::Vec2i& p, const cv::Vec2i& off,
                                  ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                  float unit, int flags)
{
    resId_t id(type, sm, p, p+off);
    if (hasResId(id))
        return 0;

    addDistLoss(sm, p, off, problem, state, unit, flags, &resId(id));

    return 1;
}

int SurfaceTracker::condDistLoss3D(int type, SurfaceMeta* sm, cv::Mat_<cv::Vec3d>& points,
                                    const cv::Vec2i& p, const cv::Vec2i& off,
                                    ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                    float unit, int flags)
{
    resId_t id(type, sm, p, p+off);
    if (hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = addDistLoss3D(points, p, off, problem, state, unit, flags, &res);

    resId(id) = res;

    return count;
}

int SurfaceTracker::condStraightLoss3D(int type, SurfaceMeta* sm, const cv::Vec2i& p,
                                        const cv::Vec2i& o1, const cv::Vec2i& o2, const cv::Vec2i& o3,
                                        cv::Mat_<cv::Vec3d>& points, ceres::Problem& problem,
                                        const cv::Mat_<uint8_t>& state, int flags)
{
    resId_t id(type, sm, p);
    if (hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = addStraightLoss3D(p, o1, o2, o3, points, problem, state, flags, &res);

    if (count)
        resId(id) = res;

    return count;
}

int SurfaceTracker::condSurfLoss(int type, SurfaceMeta* sm, const cv::Vec2i p,
                                  ceres::Problem& problem, const cv::Mat_<uint8_t>& state,
                                  cv::Mat_<cv::Vec3d>& points, float step)
{
    resId_t id(type, sm, p);
    if (hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = addSurfLoss(sm, p, problem, state, points, step, &res);

    if (count)
        resId(id) = res;

    return count;
}

// Cost and optimization functions
double SurfaceTracker::localCost(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                                  cv::Mat_<cv::Vec3d>& points, float step, float src_step,
                                  int* ref_count, int* straight_count_ptr)
{
    int count;
    int straight_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straight_count;

    double test_loss = 0.0;
    ceres::Problem problem_test;

    count = addLocal(sm, p, problem_test, state, points, step, src_step, 0, straight_count_ptr);
    if (ref_count)
        *ref_count = count;

    problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}

double SurfaceTracker::localCostDestructive(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                                             cv::Mat_<cv::Vec3d>& points, float step, float src_step,
                                             cv::Vec3f loc, int* ref_count, int* straight_count_ptr)
{
    uint8_t state_old = state(p);
    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
    int count;
    int straight_count;
    if (!straight_count_ptr)
        straight_count_ptr = &straight_count;

    double test_loss = 0.0;
    {
        ceres::Problem problem_test;

        this->loc(sm, p) = {loc[1], loc[0]};

        count = addLocal(sm, p, problem_test, state, points, step, src_step, 0, straight_count_ptr);
        if (ref_count)
            *ref_count = count;

        problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
    }
    erase(sm, p);
    state(p) = state_old;

    if (!count)
        return 0;
    else
        return sqrt(test_loss/count);
}

double SurfaceTracker::localSolve(SurfaceMeta* sm, const cv::Vec2i p, cv::Mat_<uint8_t>& state,
                                   cv::Mat_<cv::Vec3d>& points, float step, float src_step, int flags)
{
    ceres::Problem problem;

    addLocal(sm, p, problem, state, points, step, src_step, flags);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.num_residual_blocks < 3)
        return 10000;

    return summary.final_cost/summary.num_residual_blocks;
}

// Add local/global functions
int SurfaceTracker::addLocal(SurfaceMeta* sm, const cv::Vec2i p, ceres::Problem& problem,
                              const cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                              float step, float src_step, int flags, int* straight_count_ptr)
{
    int count = 0;
    int count_straight = 0;

    if (flags & LOSS_3D_INDIRECT) {
        // Direct neighbors
        count += addDistLoss3D(points, p, {0,1}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {1,0}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {0,-1}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {-1,0}, problem, state, step*src_step, flags);

        // Diagonal neighbors
        count += addDistLoss3D(points, p, {1,1}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {1,-1}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {-1,1}, problem, state, step*src_step, flags);
        count += addDistLoss3D(points, p, {-1,-1}, problem, state, step*src_step, flags);

        // Straight losses
        count_straight += addStraightLoss3D(p, {0,-2},{0,-1},{0,0}, points, problem, state);
        count_straight += addStraightLoss3D(p, {0,-1},{0,0},{0,1}, points, problem, state);
        count_straight += addStraightLoss3D(p, {0,0},{0,1},{0,2}, points, problem, state);
        count_straight += addStraightLoss3D(p, {-2,0},{-1,0},{0,0}, points, problem, state);
        count_straight += addStraightLoss3D(p, {-1,0},{0,0},{1,0}, points, problem, state);
        count_straight += addStraightLoss3D(p, {0,0},{1,0},{2,0}, points, problem, state);
    }
    else {
        // 2D losses
        count += addDistLoss(sm, p, {0,1}, problem, state, step);
        count += addDistLoss(sm, p, {1,0}, problem, state, step);
        count += addDistLoss(sm, p, {0,-1}, problem, state, step);
        count += addDistLoss(sm, p, {-1,0}, problem, state, step);
        count += addDistLoss(sm, p, {1,1}, problem, state, step);
        count += addDistLoss(sm, p, {1,-1}, problem, state, step);
        count += addDistLoss(sm, p, {-1,1}, problem, state, step);
        count += addDistLoss(sm, p, {-1,-1}, problem, state, step);

        count_straight += addStraightLoss(sm, p, {0,-2},{0,-1},{0,0}, problem, state);
        count_straight += addStraightLoss(sm, p, {0,-1},{0,0},{0,1}, problem, state);
        count_straight += addStraightLoss(sm, p, {0,0},{0,1},{0,2}, problem, state);
        count_straight += addStraightLoss(sm, p, {-2,0},{-1,0},{0,0}, problem, state);
        count_straight += addStraightLoss(sm, p, {-1,0},{0,0},{1,0}, problem, state);
        count_straight += addStraightLoss(sm, p, {0,0},{1,0},{2,0}, problem, state);
    }

    if (flags & LOSS_ZLOC)
        problem.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(
            sm->surface()->rawPoints(),
            seed_coord[2] - (p[0]-seed_loc[0])*step*src_step, z_loc_loss_w),
            new ceres::HuberLoss(1.0), &loc(sm, p)[0]);

    if (flags & SURF_LOSS) {
        count += addSurfLoss(sm, p, problem, state, points, step);
    }

    if (straight_count_ptr)
        *straight_count_ptr += count_straight;

    return count + count_straight;
}

int SurfaceTracker::addGlobal(SurfaceMeta* sm, const cv::Vec2i p, ceres::Problem& problem,
                               const cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                               float step, int flags, float step_onsurf)
{
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    int count = 0;

    if (flags & LOSS_3D_INDIRECT) {
        count += condDistLoss3D(0, sm, points, p, {0,1}, problem, state, step, flags);
        count += condDistLoss3D(0, sm, points, p, {1,0}, problem, state, step, flags);
        count += condDistLoss3D(1, sm, points, p, {0,-1}, problem, state, step, flags);
        count += condDistLoss3D(1, sm, points, p, {-1,0}, problem, state, step, flags);
        count += condDistLoss3D(2, sm, points, p, {1,1}, problem, state, step, flags);
        count += condDistLoss3D(2, sm, points, p, {1,-1}, problem, state, step, flags);
        count += condDistLoss3D(3, sm, points, p, {-1,1}, problem, state, step, flags);
        count += condDistLoss3D(3, sm, points, p, {-1,-1}, problem, state, step, flags);

        count += condStraightLoss3D(4, sm, p, {0,-2},{0,-1},{0,0}, points, problem, state, flags);
        count += condStraightLoss3D(4, sm, p, {0,-1},{0,0},{0,1}, points, problem, state, flags);
        count += condStraightLoss3D(4, sm, p, {0,0},{0,1},{0,2}, points, problem, state, flags);
        count += condStraightLoss3D(5, sm, p, {-2,0},{-1,0},{0,0}, points, problem, state, flags);
        count += condStraightLoss3D(5, sm, p, {-1,0},{0,0},{1,0}, points, problem, state, flags);
        count += condStraightLoss3D(5, sm, p, {0,0},{1,0},{2,0}, points, problem, state, flags);
        count += condStraightLoss3D(6, sm, p, {-2,-2},{-1,-1},{0,0}, points, problem, state, flags);
        count += condStraightLoss3D(6, sm, p, {-1,-1},{0,0},{1,1}, points, problem, state, flags);
        count += condStraightLoss3D(6, sm, p, {0,0},{1,1},{2,2}, points, problem, state, flags);
        count += condStraightLoss3D(7, sm, p, {-2,2},{-1,1},{0,0}, points, problem, state, flags);
        count += condStraightLoss3D(7, sm, p, {-1,1},{0,0},{1,-1}, points, problem, state, flags);
        count += condStraightLoss3D(7, sm, p, {0,0},{1,-1},{2,-2}, points, problem, state, flags);
    }

    if (flags & LOSS_ON_SURF) {
        if (step_onsurf == 0)
            throw std::runtime_error("oops step_onsurf == 0");

        count += condDistLoss(8, sm, p, {0,1}, problem, state, step_onsurf);
        count += condDistLoss(8, sm, p, {1,0}, problem, state, step_onsurf);
        count += condDistLoss(9, sm, p, {0,-1}, problem, state, step_onsurf);
        count += condDistLoss(9, sm, p, {-1,0}, problem, state, step_onsurf);
        count += condDistLoss(10, sm, p, {1,1}, problem, state, step_onsurf);
        count += condDistLoss(10, sm, p, {1,-1}, problem, state, step_onsurf);
        count += condDistLoss(11, sm, p, {-1,1}, problem, state, step_onsurf);
        count += condDistLoss(11, sm, p, {-1,-1}, problem, state, step_onsurf);
    }

    if (flags & SURF_LOSS && state(p) & STATE_LOC_VALID)
        count += condSurfLoss(14, sm, p, problem, state, points, step);

    return count;
}

// Point generation
cv::Mat_<cv::Vec3d> SurfaceTracker::genPointsHR(cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                                                  cv::Rect& used_area, float step, float step_src, bool inpaint)
{
    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(state.rows*step, state.cols*step, 0);

#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
                for(auto &sm : surfsC({j,i})) {
                    if (validInt(sm,{j,i})
                        && validInt(sm,{j,i+1})
                        && validInt(sm,{j+1,i})
                        && validInt(sm,{j+1,i+1}))
                    {
                        cv::Vec2f l00 = loc(sm,{j,i});
                        cv::Vec2f l01 = loc(sm,{j,i+1});
                        cv::Vec2f l10 = loc(sm,{j+1,i});
                        cv::Vec2f l11 = loc(sm,{j+1,i+1});

                        for(int sy=0;sy<=step;sy++)
                            for(int sx=0;sx<=step;sx++) {
                                float fx = sx/step;
                                float fy = sy/step;
                                cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                                cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                                cv::Vec2f l = (1-fy)*l0 + fy*l1;
                                if (loc_valid(sm->surface()->rawPoints(), l)) {
                                    points_hr(j*step+sy,i*step+sx) += lookupIntLoc(sm,l);
                                    counts_hr(j*step+sy,i*step+sx) += 1;
                                }
                            }
                    }
                }
                if (!counts_hr(j*step+1,i*step+1) && inpaint) {
                    cv::Vec3d c00 = points(j,i);
                    cv::Vec3d c01 = points(j,i+1);
                    cv::Vec3d c10 = points(j+1,i);
                    cv::Vec3d c11 = points(j+1,i+1);

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            if (!counts_hr(j*step+sy,i*step+sx)) {
                                float fx = sx/step;
                                float fy = sy/step;
                                cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                                cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                                cv::Vec3d c = (1-fy)*c0 + fy*c1;
                                points_hr(j*step+sy,i*step+sx) = c;
                                counts_hr(j*step+sy,i*step+sx) = 1;
                            }
                        }
                }
            }
        }

#pragma omp parallel for
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};

    return points_hr;
}


// Thread safety
SurfaceTracker* SurfaceTracker::getThreadCopy()
{
    return new SurfaceTracker(*this);
}

void SurfaceTracker::mergeThreadData(const SurfaceTracker& thread_tracker, const std::vector<cv::Vec2i>& points)
{
    std::lock_guard<std::shared_mutex> lock(mutex);
    for(auto p : points) {
        surfs(p) = thread_tracker.surfsC(p);
        for(auto& s : surfs(p)) {
            if (thread_tracker.has(s, p))
                loc(s, p) = const_cast<SurfaceTracker&>(thread_tracker).loc(s, p);
        }
    }
}

// Seed management
void SurfaceTracker::setSeed(const cv::Vec3d& coord, const cv::Vec2i& loc)
{
    seed_coord = coord;
    seed_loc = loc;
}

void SurfaceTracker::optimizeSurfaceMapping(cv::Mat_<uint8_t>& state, cv::Mat_<cv::Vec3d>& points,
                                            cv::Rect used_area, cv::Rect static_bounds,
                                            float step, float src_step, const cv::Vec2i& seed,
                                            int closing_r, bool keep_inpainted,
                                            const std::filesystem::path& tgt_dir)
{
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area << " " << static_bounds << std::endl;

    cv::Mat_<cv::Vec3d> points_new = points.clone();
    SurfaceMeta sm;
    sm._surf = new QuadSurface(points, {1,1});

    std::shared_mutex local_mutex;

    SurfaceTracker data_new;
    data_new._data = this->_data;

    used_area = cv::Rect(used_area.x-2, used_area.y-2, used_area.size().width+4, used_area.size().height+4);
    cv::Rect used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};

    ceres::Problem problem_inpaint;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
#ifdef VC_USE_CUDA_SPARSE
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
        if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
            options.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
    }
#endif
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = omp_get_max_threads();
    options.use_nonmonotonic_steps = true;

    // Setup initial surface associations
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(&sm);
                data_new.loc(&sm, {j,i}) = {j,i};
            }

    cv::Mat_<uint8_t> new_state = state.clone();

    // Generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    int res_count = 0;
    // Slowly inpaint physics only points
    for(int r=0; r<closing_r+2; r++) {
        cv::Mat_<uint8_t> masked;
        bitwise_and(state, STATE_VALID, masked);
        cv::dilate(masked, masked, m, {-1,-1}, r);
        cv::erode(masked, masked, m, {-1,-1}, std::min(r, closing_r));

        for(int j=used_area.y; j<used_area.br().y; j++)
            for(int i=used_area.x; i<used_area.br().x; i++)
                if ((masked(j,i) & STATE_VALID) && (~new_state(j,i) & STATE_VALID)) {
                    new_state(j, i) = STATE_COORD_VALID;
                    points_new(j,i) = {-3,-2,-4};
                    double err = data_new.localSolve(&sm, {j,i}, new_state, points_new, step, src_step, LOSS_3D_INDIRECT | SURF_LOSS);
                    if (points_new(j,i)[0] == -3) {
                        new_state(j, i) = 0;
                        points_new(j,i) = {-1,-1,-1};
                    }
                    else
                        res_count += data_new.addGlobal(&sm, {j,i}, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
                }
    }

    // Set constant parameters for original points
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++)
            if (state(j,i) & STATE_LOC_VALID)
                if (problem_inpaint.HasParameterBlock(&points_new(j,i)[0]))
                    problem_inpaint.SetParameterBlockConstant(&points_new(j,i)[0]);

    ceres::Solve(options, &problem_inpaint, &summary);
    std::cout << summary.BriefReport() << std::endl;

    cv::Mat_<cv::Vec3d> points_inpainted = points_new.clone();

    SurfaceMeta sm_inp;
    sm_inp._surf = new QuadSurface(points_inpainted, {1,1});

    SurfaceTracker data_inp;
    data_inp._data = data_new._data;

    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++)
            if (new_state(j,i) & STATE_LOC_VALID) {
                data_inp.surfs({j,i}).insert(&sm_inp);
                data_inp.loc(&sm_inp, {j,i}) = {j,i};
            }

    ceres::Problem problem;

    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << std::endl;

    int fix_points = 0;
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++) {
            res_count += data_inp.addGlobal(&sm_inp, {j,i}, problem, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);
            fix_points++;
            if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                problem.AddResidualBlock(LinChkDistLoss::Create(data_inp.loc(&sm_inp, {j,i}), 1.0), nullptr, &data_inp.loc(&sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: num fix points " << fix_points << std::endl;

    data_inp.setSeed(points_new(seed), seed);

    int fix_points_z = 0;
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++) {
            fix_points_z++;
            if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                problem.AddResidualBlock(ZLocationLoss<cv::Vec3d>::Create(points_new, data_inp.getSeedCoord()[2] - (j-data_inp.getSeedLoc()[0])*step*src_step, z_loc_loss_w),
                                        new ceres::HuberLoss(1.0), &data_inp.loc(&sm_inp, {j,i})[0]);
        }

    std::cout << "optimizer: optimizing " << res_count << " residuals, seed " << seed << std::endl;

    // Set static bounds as constant
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                if (problem.HasParameterBlock(&data_inp.loc(&sm_inp, {j,i})[0]))
                    problem.SetParameterBlockConstant(&data_inp.loc(&sm_inp, {j,i})[0]);
                if (problem.HasParameterBlock(&points_new(j, i)[0]))
                    problem.SetParameterBlockConstant(&points_new(j, i)[0]);
            }

    options.max_num_iterations = 1000;
    options.use_nonmonotonic_steps = true;
    options.use_inner_iterations = true;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    // Debug output
    {
        cv::Mat_<cv::Vec3d> points_hr_inp = this->genPointsHR(new_state, points_inpainted, used_area, step, src_step, true);
        try {
            QuadSurface *dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step, 1/src_step});
            std::string uuid = Z_DBG_GEN_PREFIX + get_surface_time_str() + "_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception) {
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    cv::Mat_<cv::Vec3d> points_hr = this->genPointsHR(new_state, points_inpainted, used_area, step, src_step);
    SurfaceTracker data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);
    cv::Mat_<uint8_t> support_count(state.size(), 0);

#pragma omp parallel for
    for(int j=used_area.y; j<used_area.br().y; j++)
        for(int i=used_area.x; i<used_area.br().x; i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                points_out(j, i) = points(j, i);
                state_out(j, i) = state(j, i);
                local_mutex.lock();
                data_out.surfs({j,i}) = this->surfsC({j,i});
                for(auto &s : data_out.surfs({j,i}))
                    data_out.loc(s, {j,i}) = this->loc(s, {j,i});
                local_mutex.unlock();
            }
            else if (new_state(j,i) & STATE_VALID) {
                cv::Vec2d l = data_inp.loc(&sm_inp, {j,i});
                int y = l[0];
                int x = l[1];
                l *= step;
                if (loc_valid(points_hr, l)) {
                    int src_loc_valid_count = 0;
                    if (state(y,x) & STATE_LOC_VALID) src_loc_valid_count++;
                    if (state(y,x+1) & STATE_LOC_VALID) src_loc_valid_count++;
                    if (state(y+1,x) & STATE_LOC_VALID) src_loc_valid_count++;
                    if (state(y+1,x+1) & STATE_LOC_VALID) src_loc_valid_count++;

                    support_count(j,i) = src_loc_valid_count;

                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;

                    std::set<SurfaceMeta*> surfs;
                    surfs.insert(this->surfsC({y,x}).begin(), this->surfsC({y,x}).end());
                    surfs.insert(this->surfsC({y,x+1}).begin(), this->surfsC({y,x+1}).end());
                    surfs.insert(this->surfsC({y+1,x}).begin(), this->surfsC({y+1,x}).end());
                    surfs.insert(this->surfsC({y+1,x+1}).begin(), this->surfsC({y+1,x+1}).end());

                    for(auto &s : surfs) {
                        auto *ptr = s->surface()->pointer();
                        float res = s->surface()->pointTo(ptr, points_out(j, i), same_surface_th, 10);
                        if (res <= same_surface_th) {
                            local_mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            cv::Vec3f loc = s->surface()->loc_raw(ptr);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            local_mutex.unlock();
                        }
                        delete ptr;
                    }
                }
            }

    // Filter by consistency
    for(int j=used_area.y; j<used_area.br().y-1; j++)
        for(int i=used_area.x; i<used_area.br().x-1; i++)
            if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                std::set<SurfaceMeta*> surf_src = data_out.surfs({j,i});
                for (auto s : surf_src) {
                    int count;
                    float cost = data_out.localCost(s, {j,i}, state_out, points_out, step, src_step, &count);
                    if (cost >= local_cost_inl_th) {
                        data_out.erase(s, {j,i});
                        data_out.eraseSurf(s, {j,i});
                    }
                }
            }

    cv::Mat_<uint8_t> fringe(state.size());
    cv::Mat_<uint8_t> fringe_next(state.size(), 1);
    int added = 1;

    for(int r=0; r<30 && added; r++) {
        LifeTime timer("optimizer: add iteration\n");

        fringe_next.copyTo(fringe);
        fringe_next.setTo(0);

        added = 0;
#pragma omp parallel for collapse(2) schedule(dynamic)
        for(int j=used_area.y; j<used_area.br().y-1; j++)
            for(int i=used_area.x; i<used_area.br().x-1; i++)
                if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_LOC_VALID && (fringe(j, i) || fringe_next(j, i))) {
                    local_mutex.lock_shared();
                    std::set<SurfaceMeta*> surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i}))
                        surf_cands.insert(s->overlapping.begin(), s->overlapping.end());
                    local_mutex.unlock_shared();

                    for(auto test_surf : surf_cands) {
                        local_mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            local_mutex.unlock_shared();
                            continue;
                        }
                        local_mutex.unlock_shared();

                        auto *ptr = test_surf->surface()->pointer();
                        if (test_surf->surface()->pointTo(ptr, points_out(j, i), same_surface_th, 10) > same_surface_th)
                            continue;

                        int count = 0;
                        cv::Vec3f loc_3d = test_surf->surface()->loc_raw(ptr);
                        delete ptr;
                        int straight_count = 0;
                        float cost;
                        local_mutex.lock();
                        cost = data_out.localCostDestructive(test_surf, {j,i}, state_out, points_out, step, src_step, loc_3d, &count, &straight_count);
                        local_mutex.unlock();

                        if (cost > local_cost_inl_th)
                            continue;

                        local_mutex.lock();
#pragma omp atomic
                        added++;
                        data_out.surfs({j,i}).insert(test_surf);
                        data_out.loc(test_surf, {j,i}) = {loc_3d[1], loc_3d[0]};
                        local_mutex.unlock();

                        for(int y=j-2; y<=j+2; y++)
                            for(int x=i-2; x<=i+2; x++)
                                fringe_next(y,x) = 1;
                    }
                }
        std::cout << "optimizer: added " << added << std::endl;
    }

    // Reset unsupported points
#pragma omp parallel for
    for(int j=used_area.y; j<used_area.br().y-1; j++)
        for(int i=used_area.x; i<used_area.br().x-1; i++)
            if (!static_bounds.contains(cv::Point(i,j))) {
                if (state_out(j,i) & STATE_LOC_VALID) {
                    if (data_out.surfs({j,i}).size() < 1) {
                        state_out(j,i) = 0;
                        points_out(j, i) = {-1,-1,-1};
                    }
                }
                else {
                    state_out(j,i) = 0;
                    points_out(j, i) = {-1,-1,-1};
                }
            }

    // Update this tracker with results
    points = points_out;
    state = state_out;
    *this = data_out;
    this->setSeed(points(seed), seed);

    // Final debug output
    {
        cv::Mat_<cv::Vec3d> points_hr_inp = this->genPointsHR(state, points, used_area, step, src_step, true);
        try {
            QuadSurface *dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step, 1/src_step});
            std::string uuid = Z_DBG_GEN_PREFIX + get_surface_time_str() + "_opt_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            delete dbg_surf;
        } catch (cv::Exception) {
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }
}