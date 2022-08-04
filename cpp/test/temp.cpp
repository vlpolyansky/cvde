
#include "cnpy.h"

#include <unordered_map>
#include <unordered_set>
#include <queue>

template<class K, class V>
using map = std::unordered_map<K, V>;

template<class K>
using set = std::unordered_set<K>;

using ll = long long;

template<class T>
using vec = std::vector<T>;

map<ll, set<ll>> edges;
set<ll> was;

std::queue<ll> q;
int sz;

void dfs(ll u) {
    was.insert(u);
    for (ll v : edges[u]) {
        if (was.find(v) == was.end()) {
            dfs(v);
        }
    }
}

void bfs() {
    while (!q.empty()) {
        ll u = q.front();
        sz++;
        q.pop();
        for (ll v : edges[u]) {
            if (was.find(v) == was.end()) {
                was.insert(v);
                q.push(v);
            }
        }

    }
}


int main() {
    // parameters
    std::string filename = "data/graphs/large_5d.npy";
//    int act_n = 12;
//    int switch_n = 1;
//    int act_n = 18;
//    int switch_n = 2;
//    int act_n = 30;
//    int switch_n = 4;
    int act_n = 60;
    int switch_n = 4;
    int ncomb = 1 << switch_n;

    cnpy::NpyArray data_npy = cnpy::npy_load(filename);
    int n = data_npy.shape[0];
    int w = data_npy.shape[1];
    assert(w == act_n + switch_n);

//    std::cout << "N=" << n << std::endl;


    sz = 0;
    vec<int> data = data_npy.as_vec<int>();
    for (int i = 0; i < n; i++) {
        ll outer = 0;
//        std::cout << ": " << std::endl;
        for (int j = 0; j < act_n; j++) {
            ll bit = data[i * w + j];
//            std::cout << bit;
            outer |= bit << j;
        }
//        std::cout << " " << outer << std::endl;
        for (int j = act_n; j < act_n + switch_n; j++) {
            int bit = data[i * w + j];
            assert(!(outer & (1ll << bit)));
        }
        vec<ll> vertices;
        for (int comb = 0; comb < ncomb; comb++) {
            ll inner = outer;
            for (int j = 0; j < switch_n; j++) {
                if ((comb >> j) & 1) {
                    int bit = data[i * w + act_n + j];
                    inner |= 1ll << bit;
                }
            }
            vertices.push_back(inner);
        }
        for (int a = 0; a < vertices.size(); a++) {
            for (int b = 0; b < vertices.size(); b++) {
                if (a == b) {
                    continue;
                }
                if (edges.find(vertices[a]) == edges.end()) {
                    edges[vertices[a]] = set<ll>();
                } else {
//                    std::cout << "HERE" << std::endl;
                }
                edges[vertices[a]].insert(vertices[b]);
                sz++;
//                std::cout << vertices[a] << " " << vertices[b] << std::endl;
            }
        }
    }

    std::cout << sz << std::endl;

    int n_components = 0;

    for (auto const &x : edges) {
        if (was.find(x.first) == was.end()) {
            n_components++;
            sz = 1;
            q.push(x.first);
//            dfs(x.first);
            was.insert(x.first);
            bfs();
            std::cout << "size: " << sz << std::endl;
        }
    }

    std::cout << "Number of components: " << n_components << std::endl;

    return 0;
}