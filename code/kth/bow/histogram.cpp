#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
using namespace std;
using namespace arma;


int main(int argc, char **argv){
    wall_clock tm;
    string file_path = argv[1];
    string means_file = argv[2];
    string hist_file = argv[3];
    int n_videos = atoi(argv[4]);

    mat C;
    C.load(means_file, auto_detect);
    int K = C.n_rows;
    int dim = C.n_cols;
    umat hist(n_videos, K);
    hist.zeros();

    vector<mat> V(n_videos);

    tm.tic();
    for(int v=1; v<=n_videos; v++){
        V[v-1].load(file_path+"/"+to_string(v)+".txt", auto_detect);
    }
    cout << "Time taken to load data: " << tm.toc() << " seconds."<< endl;

    tm.tic();
    for(int v=0; v<n_videos; v++){
        int N = V[v].n_rows;
        for(int n=0; n<N; n++){
            int idx=-1;
            double mi=100000000;
            for(int k=0;k<K;k++){
                double sum = 0;
                for(int d=0; d<dim; d++){
                    sum += (V[v](n,d) - C(k,d))*(V[v](n,d) - C(k,d));
                }
                // vec x = X.row(n);
                // vec y = C.row(k);
                // // cout << x.size() << " " << y.size() << endl;
                // // return 0;
                // vec diff = x-y;
                // vec d = diff.t() * diff;
                // if(d(0)<mi){
                //     mi=d(0);
                //     idx=k;
                // }
                if(sum<mi){
                    mi=sum;
                    idx=k;
                }
            }
            hist(v,idx)++;
        }
    }
    cout << "(Hist Gen Time)Time taken to generate features = " << tm.toc() << " seconds." << endl;
    hist.save(hist_file, csv_ascii);
    cout << "Hist size = " << hist.n_rows << " by " << hist.n_cols << endl;
    cout << "Finished" << endl;
    return 0;
}
