import matplotlib.pylab as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import show
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score


def CrossValScore(all_time_series, labels, root_path):
    n_splits = 5
    # 固定random_state后，每次构建的模型是相同的、生成的数据集是相同的、每次的拆分结果也是相同的。
    cv = StratifiedKFold(n_splits=n_splits)
    names = ["Neural Net"]
    classifiers = [
        MLPClassifier(hidden_layer_sizes=(1000, 500), max_iter=1000, random_state=38, early_stopping=True)]
    kinds = ['correlation', 'partial correlation', 'tangent']
    scores = {}
    for kind in kinds:
        scores[kind] = []
        for name, clf in zip(names, classifiers):
            print("name:{},classifiers:{}".format(name, clf))
            connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
            connectivity_biomarkers = connectivity.fit_transform(all_time_series)
            cv_scores = cross_val_score(clf,
                                        connectivity_biomarkers,
                                        y=labels,
                                        cv=cv,
                                        scoring='accuracy',
                                        )
            ypred_all = []  # 记录每一折测试集预测结果
            ytrue_all = []  # 记录每一折测试集真实标签
            predict_score = []  # 记录每一折测试集预测精度
            i = 0
            for train_index, test_index in cv.split(connectivity_biomarkers, labels):
                ypred_all.append([])
                ytrue_all.append([])
                new_train_num, new_test_num = connectivity_biomarkers[train_index], connectivity_biomarkers[test_index]
                new_label_train, new_label_test = labels[train_index], labels[test_index]
                print("iteration", i+1, ":")
                print('len(train set):{} \n'.format(train_index.size))
                print('train set:{} \n'.format(new_train_num))
                print('len(train label):{} \n'.format(new_label_train.size))
                print('train label set:{} \n'.format(new_label_train))

                print('len(test set):{} \n'.format(test_index.size))
                print('test set:{} \n'.format(new_test_num))
                print('len(test label set):{} \n'.format(new_label_test.size))
                print('test label set:{} \n'.format(new_label_test))

                clf.fit(new_train_num, new_label_train)
                ypred = clf.predict(new_test_num)
                print("predicted labels for data of indices", test_index, "are:", ypred)
                print("true label for data of indices", test_index, "are:", new_label_test)
                ytrue_all[i].append(new_label_test)
                ypred_all[i].append(ypred)
                for x, y in zip(ytrue_all[i], ypred_all[i]):
                    print('第' + str(i + 1) + '折的预测结果:', ypred_all[i])
                    print('第' + str(i + 1) + '折的预测标签:', ytrue_all[i])
                    print('第' + str(i+1) + '折的精度:', accuracy_score(x, y))
                    predict_score.append(accuracy_score(x, y))
                print("=====================================")
                i = i + 1
            score_predict = np.mean(predict_score)
            scores[kind].append(cv_scores)
        print("kind:{},cv_scores:{}".format(kind, scores[kind]))

        print("classifiers:{},cv_scores:{}\n".format(name, np.mean(scores[kind])))
        print("score_predict:{}".format(score_predict))
    mean_scores = [np.mean(scores[kind]) for kind in kinds]
    print("mean_scores", mean_scores)
    scores_std = [np.std(scores[kind]) for kind in kinds]
    print("scores_std", scores_std)

def main():
    all_time_series = np.load("/ADHD200Data/NYU_dataset/5_fold/Display/visiable/all_time_series.npy")
    print("all_time_series.shape", all_time_series.shape)  # (216,172,173)

    # 所有人的标签
    NYU_label = np.load('/ADHD200Data/NYU_dataset/Y_all_labels.npy', allow_pickle=True).item()
    NYU_label = sorted(NYU_label.items(), key=lambda x: x[0])
    Y_labels = []
    for i in NYU_label:
        print(i[1])
        Y_labels.append(i[1])

    print(len(Y_labels))  
    Y_labels = np.asarray(Y_labels)
    root_path = '/ADHD200Data/NYU_dataset/5_fold/Display/visiable/'
    CrossValScore(all_time_series, Y_labels, root_path)


if __name__ == "__main__":
    main()
