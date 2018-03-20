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


    double tt=0;
    int vct=0;
    int hct=0;
    while(vct!=n_videos){
        int ct=0;
        vector<mat> V(2500);
        while(ct!=2500 && vct!=n_videos){
            V[ct].load(file_path+"/"+to_string(vct+1)+".txt", auto_detect);
            //cout << vct << ":" << ct << "  ";
            ct++;
            vct++;
        }
        cout << endl;
        for(int i=0; i<2500; i++){
            if(hct==n_videos){
                break;
            }
            int Nv = (int)(V[i].n_rows);
            tm.tic();
            #pragma omp parallel for
            for(int n=0; n<Nv; n++){
                int idx=-1;
                double mi=100000000;
                for(int k=0;k<K;k++){
                    double sum = 0;
                    for(int d=0; d<dim; d++){
                        sum += (V[i](n,d) - C(k,d))*(V[i](n,d) - C(k,d));
                    }
                    if(sum<mi){
                        mi=sum;
                        idx=k;
                    }
                }
                hist(hct,idx)++;
            }
            tt+=tm.toc();
            //cout << hct << " ";
            hct++;
        }
        //cout << endl;
    }

    cout << "Time taken to generate features = " << tt << " seconds." << endl;
    hist.save(hist_file, csv_ascii);
    cout << "Hist size = " << hist.n_rows << " by " << hist.n_cols << endl;
    cout << "Finished" << endl;
    return 0;
}





///////////////////////////////////////////////////////////////////////////////////////////////////
// for(int v=1; v<=n_videos; v++){
//     mat V;
//     V.load(file_path+"/"+to_string(v)+".txt", auto_detect);
//
//     if(v%100==1){
//       cout << v << endl;
//     }
//     int Nv = (int)(V.n_rows);
//     tm.tic();
//
//     for(int n=0; n<Nv; n++){
//         int idx=-1;
//         double mi=100000000;
//         for(int k=0;k<K;k++){
//             double sum = 0;
//             for(int d=0; d<dim; d++){
//                 sum += (V(n,d) - C(k,d))*(V(n,d) - C(k,d));
//             }
//             if(sum<mi){
//                 mi=sum;
//                 idx=k;
//             }
//         }
//         hist(v-1,idx)++;
//     }
//     tt+=tm.toc();
// }
// cout << endl;
