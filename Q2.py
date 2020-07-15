from pprint import pprint
from scratch.linear_algebra import distance
from collections import Counter

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner


def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        return majority_vote(labels[:-1])  # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


#dataframe을 사용하지 못하므로 2차원배열로 불러옴
with open('data2.csv') as file:
    data2 = []
    for line in file.readlines():
        line = line.strip()
        data2.append(line.split(','))

#data2의 전체 데이터 출력
pprint(data2,width=220)


import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # generic type to represent a data point
Y = TypeVar('Y')  # generic type to represent output variables
Z = TypeVar('Z')


#기존 split_data는 랜덤으로 데이터를 섞기때문에
#저장된 순서대로 데이터를 나누기위해 새로 만드는 함수
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    #random.shuffle(data)
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

#airquality컬럼도 7:3으로 나누기위해 수정하는 함수
#반환리스트가 6개
def train_test_split_edit(xs: List[X],ys: List[Y],zs: List[Z],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y], List[Z],List[Z]]:
    # Generate the indices and split them.
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs],
            [zs[i] for i in train_idxs],
            [zs[i] for i in test_idxs])


# 각 컬럼에 따라 precision, recall, f1score 뽑아주는 함수
def set_dataset(pridict_data, str):
    print('\n\n<<{}의 경우>>'.format(str))
    '''good라고 한 경우 tp는 num_correct된 경우중 good의 숫자
                       fn은 good인데 good이 아닌걸로 예측(prrdict했지만 good이 아닌경우)
                       fp는 good이 아닌데 good이라고 pridict(good이리고 예측한 횟수에서 num_correct뺌)
                       tn는 아닌걸 아닌걸로 표현함 good이 아닌걸(correct_num중 good이 아닌갯수)'''

    # airquality에 대한 true positive값
    # str은 airquality의 이름
    tp = pridict_data.count((str, 'T'))

    # pridict_data.count(('good','F'))
    count = 0
    for i in range(len(pridict_data)):
        if pridict_data[i][1] == 'F':
            count += 1
    # 예상결과가 f인것중 예측값이 good이 아닌경우
    fn = count - pridict_data.count((str, 'F'))

    fp = pridict_data.count((str, 'F'))

    count = 0
    for i in range(len(pridict_data)):
        if pridict_data[i][1] == 'T':
            count += 1
    tn = count - pridict_data.count((str, 'T'))

    print('tp_{} :'.format(str), tp, 'tn_{} :'.format(str), tn, 'fp_{} :'.format(str), fp, 'fn_{} :'.format(str), fn)

    try:
        print('{}_precision : '.format(str), precision(tp, fp, fn, tn))
    except:
        print('TP와 FP값이 모두 0이기때문에 precision을 구할수 없습니다.')
    try:
        print('{}_recall : '.format(str), recall(tp, fp, fn, tn))
    except:
        print('TP와 FN값이 모두 0이기때문에 recall을 구할수 없습니다.')
    try:
        print('{}_f1score : '.format(str), f1_score(tp, fp, fn, tn))
    except:
        print('precision와 recall값이 모두 0이기때문에 f1score를 구할수 없습니다.')





cdate = []
scode = []
airquality = []

for i in range(1,len(data2)):
    cdate.append(data2[i][1]) #cdate컬럼

for i in range(1,len(data2)):
    scode.append(data2[i][4]) #측정소 코드

for i in range(1,len(data2)):
    airquality.append(data2[i][11]) #airquality 컬럼


cdate = list(map(int, cdate) )
scode = list(map(int, scode) )




#좌표는 scode,cdate 2개로 하고, 7:3으로 나눔 airquality도 7:3
cdate_train, cdate_test, scode_train, scode_test, air_train ,air_test = train_test_split_edit(cdate,scode,airquality,0.3)
labeled_points = list(zip(cdate_train,scode_train,air_train))
labeled_points = [([cdate, scode], airquality) for cdate, scode, airquality in labeled_points]
print('학습할 데이터의 갯수 : ',len(labeled_points))

#테스트 데이터 남은 30퍼센트
new_points = list(zip(cdate_test,scode_test,air_test))
new_points = [([cdate, scode], airquality) for cdate, scode, airquality in new_points]
print('테스트데이터 갯수 : ',len(new_points))


#훈련데이터에 대해
#최적의 k를 찾기위해 다 돌려봄
'''
for k in range(1,20,2): 
    num_correct = 0
    for location, actual_airquality in labeled_points:
        other_points = [other_point
                for other_point in labeled_points
                        if other_point != (location, actual_airquality)]
        predicted_language = knn_classify(k, other_points, location)
        if predicted_language == actual_airquality:
                        num_correct += 1
    print(k, "neighbor[s]:", num_correct, "correct out of", len(labeled_points))
    
'''
'''
*훈련데이터에 대한 k값 결과
1 neighbor[s]: 2205 correct out of 5276
3 neighbor[s]: 2359 correct out of 5276
5 neighbor[s]: 2402 correct out of 5276
7 neighbor[s]: 2326 correct out of 5276
9 neighbor[s]: 2248 correct out of 5276
11 neighbor[s]: 2263 correct out of 5276
13 neighbor[s]: 2323 correct out of 5276
15 neighbor[s]: 2340 correct out of 5276
17 neighbor[s]: 2348 correct out of 5276
19 neighbor[s]: 2412 correct out of 5276

1위는 19 2위는 5였으므로 2개의 k에 대해 더 높은값을 test데이터에 입력한다
'''

pridict_data=[]             #예측한 데이터
pridict_correct_flag=[]     #예측한 데이터가 맞았을경우 T 틀리면 F

print('k=5의 경우에 대해 구하는중...')
for k in [5]:
    num_correct = 0
    for location, actual_airquality in new_points:
        other_points = [other_point
                for other_point in new_points
                        if other_point != (location, actual_airquality)]
        predicted_airquality = knn_classify(k, other_points, location)
        pridict_data.append(predicted_airquality)
        if predicted_airquality == actual_airquality:
                        #print(predicted_language,actual_language)
                        pridict_correct_flag.append('T')
                        num_correct += 1
        else : pridict_correct_flag.append('F')
    print(k, "neighbor[s]:", num_correct, "correct out of", len(new_points))

'''
* 19와 5에대한 testdata의 결과
19 neighbor[s]: 881 correct out of 2262
5 neighbor[s]: 924 correct out of 2262

'''

from scratch.machine_learning import precision,recall,f1_score

'''good라고 한 경우 tp는 num_correct된 경우중 good의 숫자
                   fn은 good인데 good이 아닌걸로 예측(prrdict했지만 good이 아닌경우)
                   fp는 good이 아닌데 good이라고 pridict(good이리고 예측한 횟수에서 num_correct뺌)
                   tn는 아닌걸 아닌걸로 표현함 good이 아닌걸(correct_num중 good이 아닌갯수)'''

print(pridict_data)             #예측한 공기질
print(pridict_correct_flag)     #예측한 공기질이 맞았는지 확인

#위 두개를 합쳐줌
pridict_data = list(zip(pridict_data,pridict_correct_flag))



#set_dataset함수를 통해 precision과 recall과 f1-score을 뽑아줌


set_dataset(pridict_data,'best')

set_dataset(pridict_data,'better')

set_dataset(pridict_data,'good')

set_dataset(pridict_data,'normal')

set_dataset(pridict_data,'bad')

set_dataset(pridict_data,'worse')

set_dataset(pridict_data,'serious')

set_dataset(pridict_data,'worst')



