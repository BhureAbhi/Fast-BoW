database_path = /media/sda5/dataset/hmdb51
train_files_path = $(database_path)/stip_train
test_files_path = $(database_path)/stip_test
train_labels_file = $(database_path)/train-labels.txt
test_labels_file = $(database_path)/test-labels.txt
n_train_videos = 3567
n_test_videos = 1530
cluster_train_file = $(database_path)/X.txt
n_clusters = 65536
n_epochs = 100
destination_path = $(database_path)/bow/k$(n_clusters)/

cluster_means_file = $(database_path)/cluster_means/C_hmdb51_65536.txt
#cluster_labels_file = $(database_path)/Y.txt
train_hist_file = $(database_path)/train_hist.txt
test_hist_file = $(database_path)/test_hist.txt
svm_train_file = $(database_path)/train.libsvm
svm_test_file = $(database_path)/test.libsvm
svm_scale_file = $(database_path)/scale.txt
scale_svm_train_file = $(database_path)/scale_train.libsvm
scale_svm_test_file = $(database_path)/scale_test.libsvm
svm_model_file = $(scale_svm_train_file).model
pred_train_file = $(database_path)/pred_train_label.txt
pred_test_file = $(database_path)/pred_test_label.txt

CXX ?= g++
CFLAGS = -std=c++11 -O3 -fPIC
CFLAGS1 = -larmadillo -fopenmp
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale histogram csvtosvm

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
#kmeans: kmeans.cpp
#	#$(CXX) $(CFLAGS) kmeans.cpp -o kmeans $(CFLAGS1)
histogram: histogram.cpp
	$(CXX) $(CFLAGS) histogram.cpp -o histogram  $(CFLAGS1)
csvtosvm: csvtosvm
	$(CXX) $(CFLAGS) csvtosvm.cpp -o csvtosvm  $(CFLAGS1)

run: histogram csvtosvm svm-scale svm-train svm-predict
	#./kmeans $(cluster_train_file) $(cluster_means_file) $(cluster_labels_file) $(n_clusters) $(n_epochs)
	./histogram $(train_files_path) $(cluster_means_file) $(train_hist_file) $(n_train_videos)
	./histogram $(test_files_path) $(cluster_means_file) $(test_hist_file) $(n_test_videos)
	./csvtosvm $(train_hist_file) $(train_labels_file) $(svm_train_file)
	./csvtosvm $(test_hist_file) $(test_labels_file) $(svm_test_file)
	./svm-scale -l 0 -u 1 -s $(svm_scale_file) $(svm_train_file) > $(scale_svm_train_file)
	./svm-scale -r $(svm_scale_file) $(svm_test_file) > $(scale_svm_test_file)
	./svm-train -q -t 0 $(scale_svm_train_file) $(svm_model_file)
	time ./svm-predict $(scale_svm_train_file) $(svm_model_file) $(pred_train_file)
	time ./svm-predict $(scale_svm_test_file) $(svm_model_file) $(pred_test_file)

.PHONY: all run

clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale histogram csvtosvm libsvm.so.$(SHVER)

copy:
	mkdir $(destination_path)
	#cp $(cluster_means_file) $(destination_path)
	#cp $(cluster_labels_file)  $(destination_path)
	cp $(train_hist_file)  $(destination_path)
	cp $(test_hist_file)  $(destination_path)
	cp $(svm_train_file)  $(destination_path)
	cp $(svm_test_file)  $(destination_path)
	cp $(svm_scale_file)  $(destination_path)
	cp $(scale_svm_train_file)  $(destination_path)
	cp $(scale_svm_test_file)  $(destination_path)
	cp $(svm_model_file)  $(destination_path)
	cp $(pred_train_file)  $(destination_path)
	cp $(pred_test_file)  $(destination_path)
