#include <iostream>

#include "../utils.h"
#include "../RandomEngine.h"

struct edge {
    int u, v;
    ftype dst;

    edge(int u, int v, ftype dst) : u(u), v(v), dst(dst) {}
};

int n = 10000;
int d = 100;
int k = 100;
using mat = Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic>;
RandomEngine re(239);

/**
 *
 * @param T Array of edges per vertex, each sorted by distance
 * @param G Array of edges, sorted by distance
 * @return
 */
vec<edge> apply(const vec<vec<int>> &T, const vec<edge> &G, const mat &sqdist) {
    my_tqdm tqdm(G.size());
    vec<edge> result;
    for (const edge &e: G) {
        tqdm.atomic_iteration();
        bool good = true;

        for (int w: T[e.u]) {
            if (sqdist(e.u, w) > e.dst) {
                break;
            }
            if (w != e.v && sqdist(e.v, w) < e.dst) {
                good = false;
                break;
            }
        }
        if (good) {
            for (int w: T[e.v]) {
                if (sqdist(e.v, w) > e.dst) {
                    break;
                }
                if (w != e.u && sqdist(e.u, w) < e.dst) {
                    good = false;
                    break;
                }
            }
        }
        if (good) {
            result.push_back(e);
        }
    }
    tqdm.bar().finish();

    return result;
}

vec<edge> gabrielize(const vec<edge> &G, const mat &inner) {
    my_tqdm tqdm(G.size());
    vec<edge> result;
    for (const edge &e: G) {
        tqdm.atomic_iteration();
        bool good = true;
        for (int w = 0; good && w < n; w++) {
            if (w != e.u && w != e.v &&
                    inner(e.v, e.u) + inner(w, w) < inner(e.v, w) + inner(e.u, w)) {
                good = false;
            }
        }
        if (good) {
            result.push_back(e);
        }
    }
    tqdm.bar().finish();
    return result;
}

int main() {
    dmatrix data(d, n);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            data(i, j) = re.rand_float();
        }
    }

    mat sqdist(n, n);
    sqdist.setZero();
    mat inner(n, n);
    inner.setZero();

    std::cout << "Computing squared distances" << std::endl;
    my_tqdm tqdm(n);
    for (int i = 0; i < n; i++) {
        tqdm.atomic_iteration();
//        inner(i, i) = data.col(i).squaredNorm();
        for (int j = i + 1; j < n; j++) {
            sqdist(i, j) = sqdist(j, i) = (data.col(i) - data.col(j)).squaredNorm();
//            inner(i, j) = inner(j, i) = data.col(i).transpose().dot(data.col(j));
        }
    }
    tqdm.bar().finish();
    inner = data.transpose() * data;

    std::cout << "Naive implementation" << std::endl;
    tqdm = my_tqdm(n);
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        tqdm.atomic_iteration();
        for (int j = i + 1; j < n; j++) {
            bool exists = true;
            for (int k = 0; k < n; k++) {
                if (k == i || k == j) {
                    continue;
                }
                if (sqdist(i, j) > sqdist(i, k) && sqdist(i, j) > sqdist(j, k)) {
                    exists = false;
                    break;
                }
            }
            cnt += exists;
        }
    }
    tqdm.bar().finish();
    std::cout << "RNG: " << n << " " << d << " " << cnt << std::endl;

    // =============== METHOD STARTS HERE ===============
    std::cout << "RNG v2: " << n << " " << d << std::endl;
    tqdm = my_tqdm(n * (n - 1) / 2);

    vec<vec<int>> edges(n);
    mat was(n, n);
    was.setConstant(0);

    vec<edge> edges_sorted;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            edges_sorted.emplace_back(i, j, sqdist(i, j));
        }
    }
    std::sort(edges_sorted.begin(), edges_sorted.end(), [](const edge &a, const edge &b) { return a.dst < b.dst;});

//    vec<edge> new_edges_sorted;
////    vec<vec<int>> new_edges(n);
//    for (const edge &e: edges_sorted) {
//        tqdm.atomic_iteration();
//        bool good = true;
//        for (int w : edges[e.u]) {
//            if (w != e.v && sqdist(e.v, w) < e.dst) {
//                good = false;
//                break;
//            }
//        }
//        if (good) {
//            for (int w : edges[e.v]) {
//                if (w != e.u && sqdist(e.u, w) < e.dst) {
//                    good = false;
//                    break;
//                }
//            }
//        }
//        if (good) {
//            new_edges_sorted.push_back(e);
//            edges[e.u].push_back(e.v);
//            edges[e.v].push_back(e.u);
//        }
//    }
//    tqdm.bar().finish();
//    std::cout << "Supergraph size: " << new_edges_sorted.size() << std::endl;

//    vec<edge> new_edges_sorted;
////    vec<vec<int>> new_edges(n);
//    for (const edge &e: edges_sorted) {
//        tqdm.atomic_iteration();
//        if (was(e.u, e.v) == 0) {
////            cnt++;
//            new_edges_sorted.push_back(e);
//            was(e.u, e.v) = 1;
//            was(e.v, e.u) = 1;
//            for (int w: edges[e.u]) {
//                was(w, e.v) = was(e.v, w) = 1;
//            }
//            for (int w: edges[e.v]) {
//                was(w, e.u) = was(e.u, w) = 1;
//            }
//            edges[e.u].push_back(e.v);
//            edges[e.v].push_back(e.u);
//        }
//    }
//    tqdm.bar().finish();
//    edges_sorted = new_edges_sorted;
//    std::cout << "Supergraph size: " << edges_sorted.size() << std::endl;
////
//    edges_sorted = apply(edges, edges_sorted, sqdist);
//    std::cout << "Filtered by itself size: " << edges_sorted.size() << std::endl;

    std::cout << "Building knn" << std::endl;
    tqdm = my_tqdm(n);
    vec<vec<int>> knn(n);
    for (int i = 0; i < n; i++) {
        tqdm.atomic_iteration();
        vec<int> indices(n);
        for (int j = 0; j < n; j++) {
            indices[j] = j;
        }
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {return sqdist(i, a) < sqdist(i, b);});
        knn[i] = vec<int>(indices.begin() + 1, indices.begin() + 1 + k);
    }
    tqdm.bar().finish();

    edges_sorted = apply(knn, edges_sorted, sqdist);
    std::cout << "Filtered by knn size: " << edges_sorted.size() << std::endl;

//    edges_sorted = gabrielize(edges_sorted, inner);
//    std::cout << "After gabrielization size: " << edges_sorted.size() << std::endl;

//    tqdm = my_tqdm(st3_edges_sorted.size());
//    cnt = 0;
//    mat final_graph(n, n);
//    final_graph.setZero();
//    for (const edge &e: st3_edges_sorted) {
//        tqdm.atomic_iteration();
//        bool good = true;
//        for (int w = 0; good && w < n; w++) {
//            if (w != e.u && w != e.v && sqdist(e.u, w) < e.dst && sqdist(e.v, w) < e.dst) {
//                good = false;
//            }
//        }
//        if (good) {
//            cnt++;
//            final_graph(e.u, e.v) = final_graph(e.v, e.u) = 1;
//        }
//    }
//    tqdm.bar().finish();
//
//    std::cout << "Final size: " << cnt << std::endl;
    // =============== METHOD ENDS HERE ===============

//    std::cout << "Gabriel graph (naive)" << std::endl;
//    tqdm = my_tqdm(n);
//    cnt = 0;
//    for (int i = 0; i < n; i++) {
//        tqdm.atomic_iteration();
//        for (int j = i + 1; j < n; j++) {
//            bool exists = true;
//            for (int k = 0; k < n; k++) {
//                if (k == i || k == j) {
//                    continue;
//                }
//                if (inner(i, i) + inner(j, j) + 4 * inner(k, k) + 2 * inner(i, j) - 4 * inner(i, k) - 4 * inner(j, k) < 2 * sqdist(i, j)) {
//                    exists = false;
//                    break;
//                }
//            }
//            cnt += exists;
//        }
//    }
//    tqdm.bar().finish();
//    std::cout << "Gabriel " << n << " " << d << " " << cnt << std::endl;

//    {
//        dmatrix mat(3, 3);
//        mat << 1, 0, 1,
//        0, 1, 1,
//        0, 0, 1;
//        mat.col(2).normalize();
//        std::cout << mat << std::endl;
//        printf("%.5Lf\n", solid_angle(mat, 10));
//    }

//    for (int i = 0; i < 100; i++) {
//        dmatrix mat(2, 2);
//        if (i == 0) {
//            mat << 1, 0, 0, 1;
//        } else {
//            mat.setRandom();
//        }
//        mat.col(0).normalize();
//        mat.col(1).normalize();
//        ftype angle_1 = solid_angle(mat, 100) * 2 * PI;
//        ftype angle_2 = acos(mat.col(0).dot(mat.col(1)));
//        printf("%.5Lf %.5Lf\n", angle_1, angle_2);
//    }

    return 0;
}