#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
using namespace std;
using namespace arma;
//using namespace chrono;

int main(int argc, char **argv){

    string X_file = argv[1];
    string C_file = argv[2];
    string Y_file = argv[3];
    int K = atoi(argv[4]);
    int epochs = atoi(argv[5]);
    wall_clock tm;

    tm.tic();
    mat X;
    X.load(X_file, auto_detect);

    cout << "Loading time: " << tm.toc() << endl;
    cout << endl << "Train data has " << X.n_rows << " rows and " << X.n_cols << " columns." << endl;


    mat C;

    // measuring time accurate to microseconds.


    X = X.t();

    tm.tic();
    bool status = kmeans(C, X, K, random_subset, epochs, true);

    cout << "(Model Build Time) Time elapsed for kmeans clustering = " << tm.toc() << " seconds."<< endl;

    C = C.t();
    cout << "clustering successful. centers = " << C.n_rows << " by " << C.n_cols << endl;


    C.save(C_file, csv_ascii);

    X = X.t();
    int N = X.n_rows;
    int dim = X.n_cols;
    uvec Y(N);

    tm.tic();
    for(int n=0;n<N;n++){
        int idx=-1;
        double mi=100000000;
        for(int k=0;k<K;k++){
            double sum = 0;
            for(int d=0; d<dim; d++){
                sum += (X(n,d) - C(k,d))*(X(n,d) - C(k,d));
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
        Y(n) = idx+1;
    }
    cout << "Time for cluster assignment: " << tm.toc() << " seconds." << endl;
    Y.save(Y_file, csv_ascii);
    cout << "Finished." << endl;

	return 0;
}
