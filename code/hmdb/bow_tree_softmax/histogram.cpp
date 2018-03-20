#include <iostream>
#include <armadillo>
#include <string>
#include <ctime>
#include <sstream>
#include <vector>
#include <cmath>
#include <queue>
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
                sum0 += x(0,d) * temp->c0(0,d);
            }
            sum0 += temp->c0(0,dim);
            for(int d=0; d<dim; d++){
                sum1 += x(0,d) * temp->c1(0,d);
            }
            sum1 += temp->c1(0,dim);
            if(sum0>sum1){
                temp = temp->left;
            }
            else{
                temp = temp->right;
            }
        }
    }

    return -1;
}
/*
node * buildtree(mat X, int height, int epochs){
    int N = (int)(X.n_rows);
    int dim = (int)(X.n_cols);
    node *nd= new node;
    //cout << "X :"  << X.n_rows << " " << X.n_cols << endl;
    //cout<<"height ="<<" "<<height<<endl;
    if(height==0){
        nd->cluster_idx = idxcount;
        nd->left = NULL;
        nd->right = NULL;
        idxcount++;
        return nd;
    }
    mat clusters;
    nd->cluster_idx = -1;
    //cout<< "kmeans starting"<<endl;
    //bool status = kmeans(clusters, X.t(), 2, random_subset, epochs, false);
    if(status==false){
        cout << "clustering failed." << endl;
        return NULL;
    }
    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;
    clusters=clusters.t();

    //cout << "clusters: " << clusters.n_rows << "  " << clusters.n_cols << endl;

    int ct0=0, ct1=0;
    rowvec Y(N);
    for(int i=0;i<N;i++){
        double sum0 = 0;
        double sum1 =0;
        // if(i%10000==0)
        //     cout << i << endl;
        for(int d=0; d<dim; d++){
            sum0 += (X(i,d) - clusters(0,d))*(X(i,d) - clusters(0,d));
        }
        for(int d=0; d<dim; d++){
            sum1 += (X(i,d) - clusters(1,d))*(X(i,d) - clusters(1,d));
        }
        if(sum0<sum1){
            Y[i] = 0;
            ct0++;
        }
        else{
            Y[i] = 1;
            ct1++;
        }
    }

    // softmax
    double lrate = 0.1;
    double lambda = 0.1;
    int MAX_ITER = 100;

    mat W = softmax(X.t(), Y, lrate, lambda, MAX_ITER);

    nd->c0=W.row(0);
    nd->c1=W.row(1);

    mat split0(ct0, dim);
    mat split1(ct1, dim);
    ct0=0;
    ct1=0;
    for(int i=0;i<N;i++){
        // if(i%10000==0)
        //     cout << i << endl;
        if(Y[i]==0){
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
*/

int main(int argc, char **argv){

    wall_clock tm;
    string file_path = argv[1];
    string model_file = argv[2];
    string hist_file = argv[3];
    int n_videos = atoi(argv[4]);
    int K = atoi(argv[5]);
    int epochs =atoi(argv[6]);

    mat X;
    X.load(model_file, auto_detect);
    int N = (int)(X.n_rows);

    umat hist(n_videos, K);
    hist.zeros();

/////////////////////////////////////////////////////
    tm.tic();
    queue<node *> nod_queue;
    node *root = new node;
    nod_queue.push(root);
    int ndCt = 1;
    int idxcount=1;

    while(!nod_queue.empty()){
        node *curNode = nod_queue.front();
        nod_queue.pop();
        if(ndCt>=K){
            curNode->cluster_idx = idxcount;
            curNode->left = NULL;
            curNode->right = NULL;
            cout << "leaf node " << ndCt << " index " << idxcount << endl;
            idxcount++;
            ndCt++;
            continue;
        }
        curNode->c0 = X.row(2*ndCt-2);
        curNode->c1 = X.row(2*ndCt-1);
        node *lnode = new node;
        node *rnode = new node;
        curNode->left = lnode;
        curNode->right = rnode;
        nod_queue.push(lnode);
        nod_queue.push(rnode);
        curNode->cluster_idx = -1;
        cout << "in node " << ndCt << endl;
        ndCt++;
    }

    cout << "(Model Build Time) Time taken to build tree = " << tm.toc() << " seconds." << endl;
//////////////////////////////////////////////////
    mat V;
    double tt=0;
    for(int v=1; v<=n_videos; v++){
        V.load(file_path+"/"+to_string(v)+".txt", auto_detect);

        tm.tic();
        int Nv = (int)(V.n_rows);
        for(int n=0; n<Nv; n++){
            // search tree and get idx.
            int clidx = findcluster(V.row(n), root);
            hist(v-1,clidx-1)++;
        }
        tt += tm.toc();
    }
    cout << "(Hist Gen Time)Time taken to generate features/histogram = " << tt << " seconds." << endl;
    hist.save(hist_file, csv_ascii);
    cout << "Hist size = " << hist.n_rows << " by " << hist.n_cols << endl;
    cout << "Finished" << endl;

    return 0;
}
