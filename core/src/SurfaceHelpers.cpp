#include "SurfaceHelpers.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Lifetime.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/math/MathUtils.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(views, xview.hpp)

#include "SurfaceTracker.hpp"


#include <fstream>
#include <iostream>

static float space_trace_dist_w = 1.0;
float dist_th = 1.5;

struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};


static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

template <typename E>
E _max_d_ign(const E &a, const E &b)
{
    if (a == E(-1))
        return b;
    if (b == E(-1))
        return a;
    return std::max(a,b);
}

template <typename T, typename E>
void _dist_iteration(T &from, T &to, int s)
{
    E magic = -1;
#pragma omp parallel for
    for(int k=0;k<s;k++)
        for(int j=0;j<s;j++)
            for(int i=0;i<s;i++) {
                E dist = from(k,j,i);
                if (dist == magic) {
                    if (k) dist = _max_d_ign(dist, from(k-1,j,i));
                    if (k < s-1) dist = _max_d_ign(dist, from(k+1,j,i));
                    if (j) dist = _max_d_ign(dist, from(k,j-1,i));
                    if (j < s-1) dist = _max_d_ign(dist, from(k,j+1,i));
                    if (i) dist = _max_d_ign(dist, from(k,j,i-1));
                    if (i < s-1) dist = _max_d_ign(dist, from(k,j,i+1));
                    if (dist != magic)
                        to(k,j,i) = dist+1;
                    else
                        to(k,j,i) = dist;
                }
                else
                    to(k,j,i) = dist;

            }
}

template <typename T, typename E>
T distance_transform(const T &chunk, int steps, int size)
{
    T c1 = xt::empty<E>(chunk.shape());
    T c2 = xt::empty<E>(chunk.shape());

    c1 = chunk;

    E magic = -1;

    for(int n=0;n<steps/2;n++) {
        _dist_iteration<T,E>(c1,c2,size);
        _dist_iteration<T,E>(c2,c1,size);
    }

#pragma omp parallel for
    for(int z=0;z<size;z++)
        for(int y=0;y<size;y++)
            for(int x=0;x<size;x++)
                if (c1(z,y,x) == magic)
                    c1(z,y,x) = steps;

    return c1;
}


template <typename T, typename C>
int gen_space_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, float w = 0.1)
{
    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(SpaceLossAcc<T,C>::Create(t, w), nullptr, &loc(p)[0]);

    return 1;
}


template <typename T, typename C>
int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.1, float dist_th = 2)
{
    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    problem.AddResidualBlock(SpaceLineLossAcc<T,C>::Create(t, steps, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

template <typename T, typename C>
int gen_anchor_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, double *anchor, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.001)
{
    return 0;

    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    anchor[0] = loc(p+off)[0];
    anchor[1] = loc(p+off)[1];
    anchor[2] = loc(p+off)[2];

    problem.AddResidualBlock(AnchorLoss<T,C>::Create(t, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

//create all valid losses for this point
template <typename I, typename T, typename C>
int emptytrace_create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, loc, flags & OPTIMIZE_ALL);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, loc, flags & OPTIMIZE_ALL);

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {0,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        count += gen_space_loss(problem, p, state, loc, t);

        count += gen_space_line_loss(problem, p, {1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {-1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,1}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,-1}, state, loc, t, unit);
    }

    return count;
}

cv::Vec2i lower_p(const cv::Vec2i &point, const cv::Vec2i &offset)
{
    if (offset[0] == 0) {
        if (offset[1] < 0)
            return point+offset;
        else
            return point;
    }
    if (offset[0] < 0)
        return point+offset;
    else
        return point;
}

bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status)
{
    return loss_status(lower_p(p, off)) & (1 << bit);
}

int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set)
{
    if (set)
        loss_status(lower_p(p, off)) |= (1 << bit);
    return set;
}

int conditional_dist_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, float unit, bool optimize_all, float w = 1.0)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, state, out, unit, optimize_all, nullptr, w));
    return set;
};

int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, bool optimize_all)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, state, out, optimize_all));
    return set;
};

template <typename T, typename C>
int conditional_spaceline_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss(problem, p, off, state, loc, t, steps));
    return set;
};

template <typename T, typename C>
int conditional_anchor_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, double *anchor, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_anchor_loss(problem, p, off, anchor, state, loc, t, steps));
    return set;
};

//create only missing losses so we can optimize the whole problem
template <typename I, typename T, typename C>
int emptytrace_create_missing_centered_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4,
    const I &interp, Chunked3d<T,C> &t, float unit, int flags = SPACE_LOSS | OPTIMIZE_ALL)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    // if (flags & SPACE_LOSS) {
        count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, state, loc, flags);

        //vertical
        count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, state, loc, flags);
    // }

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        if (!loss_mask(6, p, {0,0}, loss_status))
            count += set_loss_mask(6, p, {0,0}, loss_status, gen_space_loss(problem, p, state, loc, t));

        count += conditional_spaceline_loss(7, p, {1,0}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(7, p, {-1,0}, loss_status, problem, state, loc, t, unit);

        count += conditional_spaceline_loss(8, p, {0,1}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(8, p, {0,-1}, loss_status, problem, state, loc, t, unit);

        //FIXME should anchor loss be from last good position (potentially of the anchors anchor?)
        count += conditional_anchor_loss(9, p, {1,0}, &a1(p)[0], loss_status, problem, state, loc, t, unit);
        count += conditional_anchor_loss(10, p, {-1,0}, &a2(p)[0], loss_status, problem, state, loc, t, unit);

        count += conditional_anchor_loss(11, p, {0,1}, &a3(p)[0], loss_status, problem, state, loc, t, unit);
        count += conditional_anchor_loss(12, p, {0,-1}, &a4(p)[0], loss_status, problem, state, loc, t, unit);
    }

    return count;
}


void freeze_inner_params(ceres::Problem &problem, int edge_dist, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out,
    cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
{
    cv::Mat_<float> dist(state.size());

    edge_dist = std::min(edge_dist,254);


    cv::Mat_<uint8_t> masked;
    bitwise_and(masked, (uint8_t)inner_flags, masked);


    cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);

    for(int j=0;j<dist.rows;j++)
        for(int i=0;i<dist.cols;i++) {
            if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.SetParameterBlockConstant(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.SetParameterBlockConstant(&loc(j,i)[0]);
                set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
            }
            if (dist(j,i) >= edge_dist+1 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.RemoveParameterBlock(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.RemoveParameterBlock(&loc(j,i)[0]);
                set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
            }
        }
}

struct thresholdedDistance
{
    enum {BORDER = 16};
    enum {CHUNK_SIZE = 64};
    enum {FILL_V = 0};
    enum {TH = 170};
    const std::string UNIQUE_ID_STRING = "dqk247q6vz_"+std::to_string(BORDER)+"_"+std::to_string(CHUNK_SIZE)+"_"+std::to_string(FILL_V)+"_"+std::to_string(TH);
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        T outer = xt::empty<E>(large.shape());

        int s = CHUNK_SIZE+2*BORDER;
        E magic = -1;

        int good_count = 0;

#pragma omp parallel for
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    if (large(z,y,x) < TH)
                        outer(z,y,x) = magic;
                    else {
                        good_count++;
                        outer(z,y,x) = 0;
                    }

        outer = distance_transform<T,E>(outer, 15, s);

        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);

        auto crop_outer = view(outer, xt::range(low,high),xt::range(low,high),xt::range(low,high));

        small = crop_outer;
    }

};


//optimize within a radius, setting edge points to constant
template <typename I, typename T, typename C>
float local_optimization(int radius, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs,
    cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, const I &interp,
    Chunked3d<T,C> &t, float unit, bool quiet = false)
{
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius)
                emptytrace_create_missing_centered_losses(problem, loss_status, op, state, locs, a1,a2,a3,a4, interp, t, unit);
        }
    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&locs(op)[0]))
                problem.SetParameterBlockConstant(&locs(op)[0]);
        }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-4;
    options.use_nonmonotonic_steps = true;

//    if (problem.NumParameterBlocks() > 1) {
//        options.use_inner_iterations = true;
//    }
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
            options.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
    }
#endif
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!quiet)
        std::cout << "local solve radius " << radius << " " << summary.BriefReport() << std::endl;

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}

QuadSurface *space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, int stop_gen, float step, const std::string &cache_root, float voxelsize)
{
    LifeTime f_timer("empty space tracing\n");
    DSReader reader = {ds,scale,cache};

    //FIXME show and handle area edge!
    int w = 2*stop_gen+50;
    int h = w;
    int z = w;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);

    int x0 = w/2;
    int y0 = h/2;

    thresholdedDistance compute;

    Chunked3d<uint8_t,thresholdedDistance> proc_tensor(compute, ds, cache, cache_root);


    passTroughComputor pass;

    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);

    std::cout << "seed val " << origin << " " <<
    (int)dbg_tensor(origin[2],origin[1],origin[0]) << std::endl;

    LifeTime *timer = new LifeTime("search & optimization ...");

    //start of empty space tracing
    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp_global(proc_tensor);

    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;

    float D = sqrt(2);

    float T = step;
    float Ts = step*reader.scale;

    int r = 1;
    int r2 = 2;


    cv::Mat_<cv::Vec3d> a1(size);
    cv::Mat_<cv::Vec3d> a2(size);
    cv::Mat_<cv::Vec3d> a3(size);
    cv::Mat_<cv::Vec3d> a4(size);

    cv::Mat_<cv::Vec3d> locs(size,cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint8_t> phys_fail(size,0);
    cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    cv::Rect used_area(x0,y0,2,2);
    //these are locations in the local volume!
    locs(y0,x0) = origin;
    locs(y0,x0+1) = origin+vx*0.1;
    locs(y0+1,x0) = origin+vy*0.1;
    locs(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

    ceres::Problem big_problem;

    int loss_count;
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0}, state, locs, a1,a2,a3,a4,interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0+1}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0+1}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);

    std::cout << "init loss count " << loss_count << std::endl;

    ceres::Solver::Options options_big;
    options_big.linear_solver_type = ceres::SPARSE_SCHUR;
    options_big.use_nonmonotonic_steps = true;
#ifdef VC_USE_CUDA_SPARSE
    // Check if Ceres was actually built with CUDA sparse support
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
        options_big.linear_solver_type = ceres::SPARSE_SCHUR;
        options_big.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;

        // Enable mixed precision for SPARSE_SCHUR
        if (options_big.linear_solver_type == ceres::SPARSE_SCHUR) {
            options_big.use_mixed_precision_solves = true;
        }
    } else {
        std::cerr << "Warning: CUDA_SPARSE requested but Ceres was not built with CUDA sparse support. Falling back to default solver." << std::endl;
    }
#endif
    options_big.minimizer_progress_to_stdout = false;
    options_big.max_num_iterations = 10000;

    ceres::Solver::Summary big_summary;
    ceres::Solve(options_big, &big_problem, &big_summary);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;


    std::cout << big_summary.BriefReport() << "\n";

    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});

    ceres::Solve(options_big, &big_problem, &big_summary);


    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    int succ = 0;

    int generation = 0;
    int phys_fail_count = 0;
    double phys_fail_th = 0.1;

    int max_local_opt_r = 4;

    std::vector<float> gen_max_cost;
    std::vector<float> gen_avg_cost;

    int ref_max = 6;
    int curr_ref_min = ref_max;

    while (fringe.size()) {
        bool global_opt = generation <= 20;

        LifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        std::vector<cv::Vec2i> rest_ps;

        for(auto p : fringe)
        {
            if ((state(p) & STATE_LOC_VALID) == 0) {
                if (state(p) & STATE_COORD_VALID)
                    for(auto n : neighs)
                        if (bounds.contains(cv::Point(p+n))
                            && (state(p+n) & (STATE_PROCESSING | STATE_LOC_VALID | STATE_COORD_VALID)) == 0) {
                            rest_ps.push_back(p+n);
                            }
                continue;
            }

            for(auto n : neighs)
                if (bounds.contains(cv::Point(p+n))
                    && (state(p+n) & STATE_PROCESSING) == 0
                    && (state(p+n) & STATE_LOC_VALID) == 0) {
                    state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        std::cout << "gen " << generation << " processing " << cands.size() << " fringe cands (total done " << succ << " fringe: " << fringe.size() << ")" << std::endl;
        fringe.resize(0);

        std::cout << "cands " << cands.size() << std::endl;

        if (generation % 10 == 0)
            curr_ref_min = std::min(curr_ref_min, 5);

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        std::vector<cv::Vec2i> thread_ps(omp_get_max_threads());

        OmpThreadPointCol cands_threadcol(max_local_opt_r*2+1, cands);

#pragma omp parallel
        {
            CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
//             int idx = rand() % cands.size();
            while (true) {
                cv::Vec2i p = cands_threadcol.next();
                if (p[0] == -1)
                    break;

                if (state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                    continue;

                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                std::vector<cv::Vec2i> srcs;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,locs.cols-1);ox++)
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count++;
                            avg += locs(oy,ox);
                            srcs.push_back({oy,ox});
                        }

                cv::Vec2i best_l = srcs[0];
                int best_ref_l = -1;
                int rec_ref_sum = 0;
                for(cv::Vec2i l : srcs) {
                    int ref_l = 0;
                    for(int oy=std::max(l[0]-r,0);oy<=std::min(l[0]+r,locs.rows-1);oy++)
                        for(int ox=std::max(l[1]-r,0);ox<=std::min(l[1]+r,locs.cols-1);ox++)
                            if (state(oy,ox) & STATE_LOC_VALID)
                                ref_l++;

                    rec_ref_sum += ref_l;

                    if (ref_l > best_ref_l) {
                        best_l = l;
                        best_ref_l = ref_l;
                    }
                }

                int ref_count2 = 0;
                for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                        // if (state(oy,ox) & (STATE_LOC_VALID | STATE_COORD_VALID)) {
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count2++;
                        }

                if (ref_count < 2 || ref_count+0.35*rec_ref_sum < curr_ref_min /*|| (generation > 3 && ref_count2 < 14)*/) {
                    state(p) &= ~STATE_PROCESSING;
#pragma omp critical
                    rest_ps.push_back(p);
                    continue;
                }

                avg /= ref_count;
                cv::Vec3d init = locs(best_l)+cv::Vec3d((rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05);
                locs(p) = init;

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, proc_tensor, Ts);

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                double loss1 = summary.final_cost;

                if (loss1 > phys_fail_th) {
                    cv::Vec3d best_loc = locs(p);
                    double best_loss = loss1;
                    for (int n=0;n<100;n++) {
                        int range = step*10;
                        locs(p) = avg + cv::Vec3d((rand()%(range*2))-range,(rand()%(range*2))-range,(rand()%(range*2))-range);
                        ceres::Solve(options, &problem, &summary);
                        loss1 = summary.final_cost;
                        if (loss1 < best_loss) {
                            best_loss = loss1;
                            best_loc = locs(p);
                        }
                        if (loss1 < phys_fail_th)
                            break;
                    }
                    loss1 = best_loss;
                    locs(p) = best_loc;
                }

                cv::Vec3d phys_only_loc = locs(p);
                // locs(p) = init;

                gen_space_loss(problem, p, state, locs, proc_tensor);

                gen_space_line_loss(problem, p, {1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {-1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,1}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,-1}, state, locs, proc_tensor, T, 0.1, 100);

                double anchors[12];
                gen_anchor_loss(problem, p, {1,0}, anchors, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {-1,0}, anchors+3, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {0,1}, anchors+6, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {0,-1}, anchors+9, state, locs, proc_tensor, T);

                ceres::Solve(options, &problem, &summary);

                double dist;
                interp.Evaluate(locs(p)[2],locs(p)[1],locs(p)[0], &dist);
                int count = 0;
                for (auto &off : neighs) {
                    if (state(p+off) & STATE_LOC_VALID) {
                        for(int i=1;i<T;i++) {
                            float f1 = float(i)/T;
                            float f2 = 1-f1;
                            cv::Vec3d l = locs(p)*f1 + locs(p+off)*f2;
                            double d2;
                            interp.Evaluate(l[2],l[1],l[0], &d2);
                            dist = std::max(dist, d2);
                            count++;
                        }
                    }
                }

                init_dist(p) = dist;

                if (dist >= dist_th || summary.final_cost >= 0.1) {
                    locs(p) = phys_only_loc;
                    state(p) = STATE_COORD_VALID;
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, a1,a2,a3,a4,
                                                                                interp_global, proc_tensor, Ts, OPTIMIZE_ALL);
                    }
                    if (loss1 > phys_fail_th) {
                        phys_fail(p) = 1;

                        float err = 0;
                        for(int range = 1; range<=max_local_opt_r;range++) {
                            err = local_optimization(range, p, state, locs, a1,a2,a3,a4, interp, proc_tensor, Ts);
                            if (err <= phys_fail_th)
                                break;
                        }
                        if (err > phys_fail_th) {
                            std::cout << "local phys fail! " << err << std::endl;
#pragma omp atomic
                            phys_fail_count++;
#pragma omp atomic
                            phys_fail_count_gen++;
                        }
                    }
                }
                else {
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, a1,a2,a3,a4,
                                                                                interp_global, proc_tensor, Ts);
                    }
#pragma omp atomic
                    succ++;
#pragma omp atomic
                    succ_gen++;
#pragma omp critical
                    {
                        if (!used_area.contains(cv::Point(p[1],p[0]))) {
                            used_area = used_area | cv::Rect(p[1],p[0],1,1);
                        }
                    }

#pragma omp critical
                    {
                        fringe.push_back(p);
                        succ_gen_ps.push_back(p);
                    }
                }
            }
        }

        if (!fringe.size() && curr_ref_min > 2) {
            curr_ref_min--;
            std::cout << used_area << std::endl;
            for(int j=used_area.y;j<=used_area.br().y;j++)
                for(int i=used_area.x;i<=used_area.br().y;i++) {
                    if (state(j, i) & STATE_LOC_VALID)
                        fringe.push_back({j,i});
                }
            std::cout << "new limit " << curr_ref_min << " " << fringe.size() << std::endl;
        }
        else if (fringe.size())
            curr_ref_min = ref_max;


        for(auto p: fringe)
            if (locs(p)[0] == -1)
                std::cout << "impossible! " << p << " " << cv::Vec2i(y0,x0) << std::endl;


        if (generation >= 3) {
            options_big.max_num_iterations = 10;
        }

        //this actually did work (but was slow ...)
        if (phys_fail_count_gen) {
            options_big.minimizer_progress_to_stdout = true;
            options_big.max_num_iterations = 100;
        }
        else
            options_big.minimizer_progress_to_stdout = false;

        if (!global_opt) {
            std::vector<cv::Vec2i> opt_local;
            for(auto p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            if (opt_local.size()) {
                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, state, locs, a1,a2,a3,a4, interp, proc_tensor, Ts, true);
                }
            }
        }
        else {
            std::cout << "running big solve" << std::endl;
            ceres::Solve(options_big, &big_problem, &big_summary);
            std::cout << big_summary.BriefReport() << "\n";
            std::cout << "avg err: " << sqrt(big_summary.final_cost/big_summary.num_residual_blocks) << std::endl;
        }

        cv::Rect used_plus = {used_area.x-8,used_area.y-8,used_area.width+16,used_area.height+16};

        if (generation > 10 && global_opt) {
            cv::Mat_<cv::Vec2d> _empty;
            freeze_inner_params(big_problem, 10, state, locs, _empty, loss_status, STATE_LOC_VALID | STATE_COORD_VALID);
        }

        cands.resize(0);

        cv::Mat_<cv::Vec3d> locs_crop = locs(used_area);
        cv::Mat_<uint8_t> state_crop = state(used_area);
        double max_cost = 0;
        double avg_cost = 0;
        int cost_count = 0;
        for(int j=0;j<locs_crop.rows;j++)
            for(int i=0;i<locs_crop.cols;i++) {
                ceres::Problem problem;
                emptytrace_create_centered_losses(problem, {j,i}, state_crop, locs_crop, interp_global, proc_tensor, Ts);
                double cost = 0.0;
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
                max_cost = std::max(max_cost, cost);
                avg_cost += cost;
                cost_count++;
            }
        gen_avg_cost.push_back(avg_cost/cost_count);
        gen_max_cost.push_back(max_cost);

        float const current_area_vx2 = double(succ)*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("-> total done %d/ fringe: %ld surf: %fG vx^2 (%f cm^2)\n", succ, (long)fringe.size(), current_area_vx2/1e9, current_area_cm2);

        timer_gen.unit = succ_gen*step*step;
        timer_gen.unit_string = "vx^2";
        print_accessor_stats();

    }
    delete timer;

    locs = locs(used_area);
    state = state(used_area);

    double max_cost = 0;
    double avg_cost = 0;
    int count = 0;
    for(int j=0;j<locs.rows;j++)
        for(int i=0;i<locs.cols;i++) {
            ceres::Problem problem;
            emptytrace_create_centered_losses(problem, {j,i}, state, locs, interp_global, proc_tensor, Ts);
            double cost = 0.0;
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
            max_cost = std::max(max_cost, cost);
            avg_cost += cost;
            count++;
        }
    avg_cost /= count;

    float const area_est_vx2 = succ*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    printf("generated approximate surface %f vx^2 (%f cm^2)\n", area_est_vx2, area_est_cm2);

    QuadSurface *surf = new QuadSurface(locs, {1/T, 1/T});

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_vx2"] = area_est_vx2;
    (*surf->meta)["area_cm2"] = area_est_cm2;
    (*surf->meta)["max_cost"] = max_cost;
    (*surf->meta)["avg_cost"] = avg_cost;
    (*surf->meta)["max_gen"] = generation;
    (*surf->meta)["gen_avg_cost"] = gen_avg_cost;
    (*surf->meta)["gen_max_cost"] = gen_max_cost;
    (*surf->meta)["seed"] = {origin[0],origin[1],origin[2]};
    (*surf->meta)["elapsed_time_s"] = f_timer.seconds();

    return surf;
}