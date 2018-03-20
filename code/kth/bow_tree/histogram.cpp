#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;
using namespace arma;

// function  tree
struct node{
    int cluster_idx;
    mat c0;
    mat c1;
    node *left;
    node *right;
};

int aaa;
int idxcount=1;

int findcluster(mat x, node * root){
    int dim = (int)(x.n_cols);
    node * temp=root;
    while(true){
        if(temp->left==NULL && temp->right==NULL){
            return temp->cluster_idx;
        }
        else{
            double sum0 = 0;
            double sum1 =0;
            for(int d=0; d<dim; d++){
                sum0 += (x(0,d) - temp->c0(0,d))*(x(0,d) - temp->c0(0,d));
            }
            for(int d=0; d<dim; d++){
                sum1 += (x(0,d) - temp->c1(0,d))*(x(0,d) - temp->c1(0,d));
            }
            if(sum0<sum1){
                temp = temp->left;
            }
            else{
                temp = temp->right;
            }
        }
    }

    return -1;
}

node * buildtree(mat X, int height, int epochs){
    int N = (int)(X.n_rows);
    int dim = (int)(X.n_cols);
    node *nd= new node;
    //cout << "X :"  << X.n_rows << " " << X.n_cols << endl;
    //cout<<"height ="<<" "<<height<<endl;
    if(height==0 || N<2){
        nd->cluster_idx = idxcount;
        nd->left = NULL;
        nd->right = NULL;
        idxcount++;
        return nd;
    }
    mat clusters;
    nd->cluster_idx = -1;
    //cout<< "kmeans starting"<<endl;
    bool status = kmeans(clusters, X.t(), 2, random_subset, epochs, false);
    if(status==false){
        cout << "clustering failed." << endl;
        return NULL;
    }
    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;
    clusters=clusters.t();
    nd->c0=clusters.row(0);
    nd->c1=clusters.row(1);
    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;

    int ct0=0, ct1=0;
    vector<int> res(N);
    for(int i=0;i<N;i++){
        double sum0 = 0;
        double sum1 =0;
        // if(i%10000==0)
        //     cout << i << endl;
        for(int d=0; d<dim; d++){
            sum0 += (X(i,d) - (nd->c0)(0,d))*(X(i,d) - (nd->c0)(0,d));
        }
        for(int d=0; d<dim; d++){
            sum1 += (X(i,d) - (nd->c1)(0,d))*(X(i,d) - (nd->c1)(0,d));
        }
        if(sum0<sum1){
            res[i] = 0;
            ct0++;
        }
        else{
            res[i] = 1;
            ct1++;
        }
    }
    mat split0(ct0, dim);
    mat split1(ct1, dim);
    ct0=0;
    ct1=0;
    for(int i=0;i<N;i++){
        // if(i%10000==0)
        //     cout << i << endl;
        if(res[i]==0){
            split0.row(ct0) = X.row(i);
            ct0++;
        }
        else{
            split1.row(ct1) = X.row(i);
            ct1++;
        }
    }

    //cout << "split0: " << split0.n_rows << "  " << split0.n_cols << endl;
    //cout << "split1: " << split1.n_rows << "  " << split1.n_cols << endl << endl;

    nd->left = buildtree(split0, height-1, epochs);
    nd->right = buildtree(split1, height-1, epochs);

    return nd;
}

int main(int argc, char **argv){

    wall_clock tm;
    string file_path = argv[1];
    string X_file = argv[2];
    string hist_file = argv[3];
    int n_videos = atoi(argv[4]);
    int K = atoi(argv[5]);
    int epochs =atoi(argv[6]);

    mat X;
    X.load(X_file, auto_detect);
    int N = (int)(X.n_rows);


    //mat idx(1, K);
    // for(int i=0; i<K; i++){
    //     idx(0,i) = i;
    // }
    umat hist(n_videos, K);
    hist.zeros();

    vector<mat> V(n_videos);

    tm.tic();
    for(int v=1; v<=n_videos; v++){
        V[v-1].load(file_path+"/"+to_string(v)+".txt", auto_detect);
    }
    cout << "Time taken to load video data: " << tm.toc() << " seconds."<< endl;

    tm.tic();
    // build tree from C.
    int height = (int)(log2(K));
    node *root= new node;
    root=buildtree(X, height, epochs);
    cout << "(Model Build Time) Time taken to build tree = " << tm.toc() << " seconds." << endl;

    tm.tic();
    for(int v=0; v<n_videos; v++){
        int Nv = (int)(V[v].n_rows);
        for(int n=0; n<Nv; n++){

            // search tree and get idx.
            int clidx = findcluster(V[v].row(n), root);
            hist(v,clidx-1)++;
        }
    }
    cout << "(Hist Gen Time)Time taken to generate features/histogram = " << tm.toc() << " seconds." << endl;
    hist.save(hist_file, csv_ascii);
    cout << "Hist size = " << hist.n_rows << " by " << hist.n_cols << endl;
    cout << "Finished" << endl;

    return 0;
}
