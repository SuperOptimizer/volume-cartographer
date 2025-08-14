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
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/math/MathUtils.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(views, xview.hpp)

#include "SurfaceTracker.hpp"


#include <fstream>
#include <iostream>

struct resId_hash {
    size_t operator()(SurfaceTracker::resId_t id) const
    {
        size_t hash1 = std::hash<int>{}(id._type);
        size_t hash2 = std::hash<void*>{}(id._sm);
        size_t hash3 = std::hash<int>{}(id._p[0]);
        size_t hash4 = std::hash<int>{}(id._p[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};


struct SurfPoint_hash {
    size_t operator()(SurfaceTracker::SurfPoint p) const
    {
        size_t hash1 = std::hash<void*>{}(p.first);
        size_t hash2 = std::hash<int>{}(p.second[0]);
        size_t hash3 = std::hash<int>{}(p.second[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

QuadSurface *grow_surf_from_surfs(SurfaceMeta *seed, const std::vector<SurfaceMeta*> &surfs_v, const nlohmann::json &params, float voxelsize)
{
    bool flip_x = params.value("flip_x", 0);
    int global_steps_per_window = params.value("global_steps_per_window", 0);

    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "flip_x: " << flip_x << std::endl;
    std::filesystem::path tgt_dir = params["tgt_dir"];

    std::unordered_map<std::string,SurfaceMeta*> surfs;
    float src_step = params.value("src_step", 20);
    float step = params.value("step", 10);
    int max_width = params.value("max_width", 80000);

    // Create SurfaceTracker with parameters from JSON
    SurfaceTracker tracker(params);

    // Print configuration values from tracker
    std::cout << "  local_cost_inl_th: " << tracker.getLocalCostInlTh() << std::endl;
    std::cout << "  same_surface_th: " << tracker.getSameSurfaceTh() << std::endl;
    // Other parameters are printed inside the constructor if needed

    std::cout << "total surface count: " << surfs_v.size() << std::endl;

    std::set<SurfaceMeta*> approved_sm;

    for(auto &sm : surfs_v) {
        if (sm->meta->contains("tags") && sm->meta->at("tags").contains("approved"))
            approved_sm.insert(sm);
        if (!sm->meta->contains("tags") || !sm->meta->at("tags").contains("defective")) {
            surfs[sm->name()] = sm;
        }
    }

    for(auto sm : approved_sm)
        std::cout << "approved: " << sm->name() << std::endl;

    for(auto &sm : surfs_v)
        for(auto name : sm->overlapping_str)
            if (surfs.count(name))
                sm->overlapping.insert(surfs[name]);

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << seed->name() << " seed overlapping: "
              << seed->overlapping.size() << "/" << seed->overlapping_str.size() << std::endl;

    cv::Mat_<cv::Vec3f> seed_points = seed->surface()->rawPoints();

    int stop_gen = 100000;
    int closing_r = 20;

    // Get sliding window scale from params
    float sliding_w_scale = params.value("sliding_w_scale", 1.0f);
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    int w = 2000/src_step/step*2+10+2*closing_r;
    int h = 15000/src_step/step*2+10+2*closing_r;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    cv::Rect save_bounds_inv(closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10);
    cv::Rect active_bounds(closing_r+5,closing_r+5,w-closing_r-10,h-closing_r-10);
    cv::Rect static_bounds(0,0,0,h);

    int x0 = w/2;
    int y0 = h/2;
    int r = 1;

    std::cout << "starting with size " << size << " seed " << cv::Vec2i(y0,x0) << std::endl;

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    std::unordered_set<cv::Vec2i,vec2i_hash> fringe;

    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> inliers_sum_dbg(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});

    cv::Rect used_area(x0,y0,2,2);
    cv::Rect used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};

    cv::Vec2i seed_loc = {seed_points.rows/2, seed_points.cols/2};

    while (seed_points(seed_loc)[0] == -1) {
        seed_loc = {rand() % seed_points.rows, rand() % seed_points.cols };
        std::cout << "try loc " << seed_loc << std::endl;
    }

    tracker.loc(seed,{y0,x0}) = {seed_loc[0], seed_loc[1]};
    tracker.surfs({y0,x0}).insert(seed);
    points(y0,x0) = tracker.lookupInt(seed,{y0,x0});

    cv::Vec3d seed_coord = points(y0,x0);
    tracker.setSeed(seed_coord, cv::Vec2i(y0,x0));

    std::cout << "seed coord " << seed_coord << " at " << cv::Vec2i(y0,x0) << std::endl;

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    fringe.insert(cv::Vec2i(y0,x0));

    // Insert initial surfs per location
    for(auto p : fringe) {
        tracker.surfs(p).insert(seed);
        cv::Vec3f coord = points(p);
        std::cout << "testing " << p << " from cands: " << seed->overlapping.size() << coord << std::endl;
        for(auto s : seed->overlapping) {
            auto *ptr = s->surface()->pointer();
            if (s->surface()->pointTo(ptr, coord, tracker.getSameSurfaceTh()) <= tracker.getSameSurfaceTh()) {
                cv::Vec3f loc = s->surface()->loc_raw(ptr);
                tracker.surfs(p).insert(s);
                tracker.loc(s, p) = {loc[1], loc[0]};
            }
            delete ptr;
        }
        std::cout << "fringe point " << p << " surfcount " << tracker.surfs(p).size() << " init " << tracker.loc(seed, p) << tracker.lookupInt(seed, p) << std::endl;
    }

    std::cout << "starting from " << x0 << " " << y0 << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;

    int final_opts = global_steps_per_window;

    int loc_valid_count = 0;
    int succ = 0;
    int inlier_base_threshold = params.value("inlier_base_threshold", 20);
    int curr_best_inl_th = inlier_base_threshold;
    int last_succ_parametrization = 0;

    std::vector<SurfaceTracker*> tracker_threads(omp_get_max_threads());
    std::vector<std::vector<cv::Vec2i>> added_points_threads(omp_get_max_threads());
    for(int i=0;i<omp_get_max_threads();i++)
        tracker_threads[i] = tracker.getThreadCopy();

    bool at_right_border = false;
    for(int generation=0;generation<stop_gen;generation++) {
        std::unordered_set<cv::Vec2i,vec2i_hash> cands;
        if (generation == 0) {
            cands.insert(cv::Vec2i(y0-1,x0));
        }
        else
            for(auto p : fringe)
            {
                if ((state(p) & STATE_LOC_VALID) == 0)
                    continue;

                for(auto n : neighs) {
                    cv::Vec2i pn = p+n;
                    if (save_bounds_inv.contains(cv::Point(pn))
                        && (state(pn) & STATE_PROCESSING) == 0
                        && (state(pn) & STATE_LOC_VALID) == 0)
                    {
                        state(pn) |= STATE_PROCESSING;
                        cands.insert(pn);
                    }
                    else if (!save_bounds_inv.contains(cv::Point(pn)) && save_bounds_inv.br().y <= pn[1]) {
                        at_right_border = true;
                    }
                }
            }
            fringe.clear();

            std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << std::endl;

            OmpThreadPointCol threadcol(3, cands);

            std::shared_mutex mutex;
            int best_inliers_gen = 0;
#pragma omp parallel
        while (true)
        {
            cv::Vec2i p = threadcol.next();

            if (p[0] == -1)
                break;

            if (state(p) & STATE_LOC_VALID)
                continue;

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            std::set<SurfaceMeta*> local_surfs = {seed};

            mutex.lock_shared();
            SurfaceTracker *tracker_th = tracker_threads[omp_get_thread_num()];
            tracker_th->mergeThreadData(tracker, added_points_threads[omp_get_thread_num()]);
            mutex.unlock_shared();
            mutex.lock();
            added_points_threads[omp_get_thread_num()].resize(0);
            mutex.unlock();

            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = tracker_th->surfsC({oy,ox});
                        local_surfs.insert(p_surfs.begin(), p_surfs.end());
                    }

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            SurfaceMeta *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            bool best_ref_seed = false;
            bool best_approved = false;

            for(auto ref_surf : local_surfs) {
                int ref_count = 0;
                cv::Vec2d avg = {0,0};
                cv::Vec3d any_p = {0,0,0};
                bool ref_seed = false;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                        if ((state(oy,ox) & STATE_LOC_VALID) && tracker_th->validInt(ref_surf,{oy,ox})) {
                            ref_count++;
                            avg += tracker_th->loc(ref_surf,{oy,ox});
                            any_p = points(cv::Vec2i(tracker_th->loc(ref_surf,{oy,ox})));
                            if (tracker_th->getSeedLoc() == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;

                tracker_th->loc(ref_surf,p) = avg + cv::Vec2d((rand() % 1000)/500.0-1, (rand() % 1000)/500.0-1);

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                int straight_count_init = 0;
                int count_init = tracker_th->addLocal(ref_surf, p, problem, state, points, step, src_step, LOSS_ZLOC, &straight_count_init);
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                float cost_init = sqrt(summary.final_cost/count_init);

                bool fail = false;
                cv::Vec2d ref_loc = tracker_th->loc(ref_surf,p);

                if (!tracker_th->validInt(ref_surf,p))
                    fail = true;

                cv::Vec3d coord;

                if (!fail) {
                    coord = tracker_th->lookupInt(ref_surf,p);
                    if (coord[0] == -1)
                        fail = true;
                }

                if (fail) {
                    tracker_th->erase(ref_surf, p);
                    continue;
                }

                state(p) = 0;

                int inliers_sum = 0;
                int inliers_count = 0;

                if (approved_sm.count(ref_surf) && straight_count_init >= 2 && count_init >= 4) {
                    std::cout << "found approved sm " << ref_surf->name() << std::endl;
                    best_inliers = 1000;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                    tracker_th->erase(ref_surf, p);
                    best_approved = true;
                    break;
                }

                for(auto test_surf : local_surfs) {
                    auto *ptr = test_surf->surface()->pointer();
                    if (test_surf->surface()->pointTo(ptr, coord, tracker_th->getSameSurfaceTh(), 10) <= tracker_th->getSameSurfaceTh()) {
                        int count = 0;
                        int straight_count = 0;
                        state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        tracker_th->loc(test_surf, p) = {loc[1], loc[0]};
                        float cost = tracker_th->localCost(test_surf, p, state, points, step, src_step, &count, &straight_count);
                        state(p) = 0;
                        tracker_th->erase(test_surf, p);
                        float straight_min_count = params.value("straight_min_count", 1.0f);
                        if (cost < tracker_th->getLocalCostInlTh() && (ref_seed || (count >= 2 && straight_count >= straight_min_count))) {
                            inliers_sum += count;
                            inliers_count++;
                        }
                    }
                    delete ptr;
                }
                if ((inliers_count >= 2 || ref_seed) && inliers_sum > best_inliers) {
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                }
                tracker_th->erase(ref_surf, p);
            }

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            if (!best_approved && (best_inliers >= curr_best_inl_th || best_ref_seed))
            {
                cv::Vec2f tmp_loc_;
                cv::Rect used_th = used_area;
                float dist = pointTo(tmp_loc_, points(used_th), best_coord, tracker_th->getSameSurfaceTh(), 1000, 1.0/(step*src_step));
                tmp_loc_ += cv::Vec2f(used_th.x,used_th.y);
                if (dist <= tracker_th->getSameSurfaceTh()) {
                    int state_sum = state(tmp_loc_[1],tmp_loc_[0]) + state(tmp_loc_[1]+1,tmp_loc_[0]) + state(tmp_loc_[1],tmp_loc_[0]+1) + state(tmp_loc_[1]+1,tmp_loc_[0]+1);
                    best_inliers = -1;
                    best_ref_seed = false;
                    if (!state_sum)
                        throw std::runtime_error("this should not have any location?!");
                }
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");

                tracker_th->surfs(p).insert(best_surf);
                tracker_th->loc(best_surf, p) = best_loc;
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                points(p) = best_coord;
                inliers_sum_dbg(p) = best_inliers;

                ceres::Problem problem;
                tracker_th->addLocal(best_surf, p, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);

                std::set<SurfaceMeta*> more_local_surfs;

                for(auto test_surf : local_surfs) {
                    for(auto s : test_surf->overlapping)
                        if (!local_surfs.count(s) && s != best_surf)
                            more_local_surfs.insert(s);

                    if (test_surf == best_surf)
                        continue;

                    auto *ptr = test_surf->surface()->pointer();
                    if (test_surf->surface()->pointTo(ptr, best_coord, tracker_th->getSameSurfaceTh(), 10) <= tracker_th->getSameSurfaceTh()) {
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        tracker_th->loc(test_surf, p) = {loc[1], loc[0]};
                        int count = 0;
                        float cost = tracker_th->localCost(test_surf, p, state, points, step, src_step, &count);
                        if (cost < tracker_th->getLocalCostInlTh()) {
                            tracker_th->surfs(p).insert(test_surf);
                            tracker_th->addLocal(test_surf, p, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC);
                        }
                        else
                            tracker_th->erase(test_surf, p);
                    }
                    delete ptr;
                }

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                for(auto test_surf : more_local_surfs) {
                    auto *ptr = test_surf->surface()->pointer();
                    float res = test_surf->surface()->pointTo(ptr, best_coord, tracker_th->getSameSurfaceTh(), 10);
                    if (res <= tracker_th->getSameSurfaceTh()) {
                        cv::Vec3f loc = test_surf->surface()->loc_raw(ptr);
                        cv::Vec3f coord = tracker_th->lookupIntLoc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            continue;
                        }
                        int count = 0;
                        float cost = tracker_th->localCostDestructive(test_surf, p, state, points, step, src_step, loc, &count);
                        if (cost < tracker_th->getLocalCostInlTh()) {
                            tracker_th->loc(test_surf, p) = {loc[1], loc[0]};
                            tracker_th->surfs(p).insert(test_surf);
                        };
                    }
                    delete ptr;
                }

                mutex.lock();
                succ++;

                tracker.surfs(p) = tracker_th->surfs(p);
                for(auto &s : tracker.surfs(p))
                    if (tracker_th->has(s, p))
                        tracker.loc(s, p) = tracker_th->loc(s, p);

                for(int t=0;t<omp_get_max_threads();t++)
                    added_points_threads[t].push_back(p);

                if (!used_area.contains(cv::Point(p[1],p[0]))) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};
                }
                fringe.insert(p);
                mutex.unlock();
            }
            else if (best_inliers == -1) {
                state(p) = 0;
                points(p) = {-1,-1,-1};
            }
            else {
                state(p) = 0;
                points(p) = {-1,-1,-1};
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        }

        if (generation == 1 && flip_x) {
            tracker.flipX(x0);

            for(int i=0;i<omp_get_max_threads();i++) {
                delete tracker_threads[i];
                tracker_threads[i] = tracker.getThreadCopy();
                added_points_threads[i].clear();
            }

            cv::Mat_<uint8_t> state_orig = state.clone();
            cv::Mat_<cv::Vec3d> points_orig = points.clone();
            state.setTo(0);
            points.setTo(cv::Vec3d(-1,-1,-1));
            cv::Rect new_used_area = used_area;
            for(int j=used_area.y;j<=used_area.br().y+1;j++)
                for(int i=used_area.x;i<=used_area.br().x+1;i++)
                    if (state_orig(j, i)) {
                        int nx = x0+x0-i;
                        int ny = j;
                        state(ny, nx) = state_orig(j, i);
                        points(ny, nx) = points_orig(j, i);
                        new_used_area = new_used_area | cv::Rect(nx,ny,1,1);
                    }

            used_area = new_used_area;
            used_area_hr = {used_area.x*step, used_area.y*step, used_area.width*step, used_area.height*step};

            fringe.clear();
            for(int j=used_area.y-2;j<=used_area.br().y+2;j++)
                for(int i=used_area.x-2;i<=used_area.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
        }

        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = params.value("consensus_limit_th", 2);
        int inl_lower_bound = inl_lower_bound_reg;

        if (!at_right_border && curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;

        if (!fringe.size() && curr_best_inl_th > inl_lower_bound) {
            curr_best_inl_th -= (1+curr_best_inl_th-inl_lower_bound)/2;
            curr_best_inl_th = std::min(curr_best_inl_th, std::max(best_inliers_gen,inl_lower_bound));
            if (curr_best_inl_th >= inl_lower_bound) {
                cv::Rect active = active_bounds & used_area;
                for(int j=active.y-2;j<=active.br().y+2;j++)
                    for(int i=active.x-2;i<=active.br().x+2;i++)
                        if (state(j,i) & STATE_LOC_VALID)
                                fringe.insert(cv::Vec2i(j,i));
            }
        }
        else
            curr_best_inl_th = inlier_base_threshold;

        loc_valid_count = 0;
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    loc_valid_count++;

        bool update_mapping = (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (!fringe.size() && final_opts) {
            final_opts--;
            update_mapping = true;
        }

        if (!global_steps_per_window)
            update_mapping = false;

        if (generation % 50 == 0 || update_mapping) {
            {
                cv::Mat_<cv::Vec3d> points_hr = tracker.genPointsHR(state, points, used_area, step, src_step);
                QuadSurface *dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_est_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_est_cm2;
                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str();
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        if (update_mapping) {
            cv::Rect active = active_bounds & used_area;
            tracker.optimizeSurfaceMapping(state, points, active, static_bounds, step, src_step, {y0,x0}, closing_r, true, tgt_dir);

            if (active.area() > 0) {
                for(int i=0;i<omp_get_max_threads();i++) {
                    delete tracker_threads[i];
                    tracker_threads[i] = tracker.getThreadCopy();
                    added_points_threads[i].resize(0);
                }
            }

            last_succ_parametrization = loc_valid_count;
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));

            {
                cv::Mat_<cv::Vec3d> points_hr = tracker.genPointsHR(state, points, used_area, step, src_step);
                QuadSurface *dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = new nlohmann::json;
                (*dbg_surf->meta)["vc_grow_seg_from_segments_params"] = params;

                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_opt";
                float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
                float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
                (*dbg_surf->meta)["area_vx2"] = area_est_vx2;
                (*dbg_surf->meta)["area_cm2"] = area_est_cm2;
                dbg_surf->save(tgt_dir / uuid, uuid);
                delete dbg_surf;
            }
        }

        float const current_area_vx2 = loc_valid_count*src_step*src_step*step*step;
        float const current_area_cm2 = current_area_vx2 * voxelsize * voxelsize / 1e8;
        printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n",
               generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()),
               current_area_vx2, current_area_cm2, best_inliers_gen);

        if (!fringe.size() && w < max_width/step)
        {
            at_right_border = false;
            std::cout << "expanding by " << sliding_w << std::endl;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            final_opts = global_steps_per_window;
            w += sliding_w;
            size = {w,h};
            bounds = {0,0,w-1,h-1};
            save_bounds_inv = {closing_r+5,closing_r+5,h-closing_r-10,w-closing_r-10};

            cv::Mat_<cv::Vec3d> old_points = points;
            points = cv::Mat_<cv::Vec3d>(size, {-1,-1,-1});
            old_points.copyTo(points(cv::Rect(0,0,old_points.cols,h)));

            cv::Mat_<uint8_t> old_state = state;
            state = cv::Mat_<uint8_t>(size, 0);
            old_state.copyTo(state(cv::Rect(0,0,old_state.cols,h)));

            cv::Mat_<uint16_t> old_inliers_sum_dbg = inliers_sum_dbg;
            inliers_sum_dbg = cv::Mat_<uint8_t>(size, 0);
            old_inliers_sum_dbg.copyTo(inliers_sum_dbg(cv::Rect(0,0,old_inliers_sum_dbg.cols,h)));

            int overlap = 5;
            active_bounds = {w-sliding_w-2*closing_r-10-overlap,closing_r+5,sliding_w+2*closing_r+10+overlap,h-closing_r-10};
            static_bounds = {0,0,w-sliding_w-2*closing_r-10,h};

            cv::Rect active = active_bounds & used_area;

            std::cout << size << bounds << save_bounds_inv << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            fringe.clear();
            curr_best_inl_th = inlier_base_threshold;
            for(int j=active.y-2;j<=active.br().y+2;j++)
                for(int i=active.x-2;i<=active.br().x+2;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert(cv::Vec2i(j,i));
        }

        cv::imwrite(tgt_dir / "inliers_sum.tif", inliers_sum_dbg(used_area));

        if (!fringe.size())
            break;
    }

    // Clean up thread trackers
    for(int i=0;i<omp_get_max_threads();i++)
        delete tracker_threads[i];

    float const area_est_vx2 = loc_valid_count*src_step*src_step*step*step;
    float const area_est_cm2 = area_est_vx2 * voxelsize * voxelsize / 1e8;
    std::cout << "area est: " << area_est_vx2 << " vx^2 (" << area_est_cm2 << " cm^2)" << std::endl;

    cv::Mat_<cv::Vec3d> points_hr = tracker.genPointsHR(state, points, used_area, step, src_step);

    QuadSurface *surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_vx2"] = area_est_vx2;
    (*surf->meta)["area_cm2"] = area_est_cm2;

    return surf;
}