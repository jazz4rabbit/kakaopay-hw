```python
def print_title(title, back_ch='-', width=80, end='\n'):
    """title을 멋지게(?) 프린트해주는 함수
    
    Parameters
    ----------
    title: str, 제목이나 소제목
    back_ch: str, optional, 제목 주위에 표시될 문자
    width: int, optional, 제목까지 합쳐서 표시될 프린트 문자열의 너비
    
    """


def csv2arff(fname_csv, header):
    """arff header 정보로 csv로 부터 csv와 동일한 경로에 arff 파일을 생성한다.
    e.g. /data/path/to/iris.csv로 부터 /data/path/to/iris.arff 파일 생성
    
    Parameters
    ----------
    fname_csv : str, csv filename
    header : str, arff 파일을 만들기 위한 헤더 정보[1].
    
    References
    ----------
    .. [1] `WEKA, Attribute-Relation File Format (ARFF), 
           <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_
    """


def load_data(dfile):
    """arff 형식의 데이터 파일경로를 입력받아 Weka API의 사용할 data 클래스를 반환한다.
    Parameters
    ----------
    dfile: str, 데이터 파일경로, arff 형식의 데이터
    
    Note
    ----------
    예측 클래스를 마지막 Attribute(열, column)으로 설정한다.
    `jvm`이 시작 상태여야 한다.
    
    Return
    ----------
    data: Weka API를 사용할 수 있는 data 클래스
    """


def save_train_test_split(dfile, percentage, rng=None):
    """dfile(arff 형식의 데이터 파일경로)를 입력받아, train과 test 데이터를 분리하여
    dfile과 동일한 경로에 train.arff와 test.arff 파일을 생성한다.
    e.g. /data/path/to/iris.arff로 부터 /data/path/to/iris_train.arff와
         /data/path/to/iris_test.arff 파일 생성
    
    Parameters
    ----------
    dfile: str, 데이터 파일경로, arff 형식의 데이터
    
    Note
    ----------
    `jvm`이 시작 상태여야 한다.
    """
    
    
def scatter_plots(dfile, outpath='./assets/scatter_plot/'):
    """특징간의 산점도를 그린다.
    
    Parameters
    ----------
    dfile: str, 데이터 파일경로, arff 형식의 데이터
    output: 산점도 그림(svg 파일)이 출력 될 경로
    
    Note
    ----------
    `jvm`이 시작 상태여야 한다.
    outpath의 폴더가 생성된다.
    """    


def feature_selection(
    data, asSearch, asEvaluator, search_option=None, evaluator_option=None, vervose=False):
    """weka형식의 데이터를 입력받아, 특징의 탐색 방법과 평가 방법을 입력받아 알고리즘을 수행후에 특징을 반환한다.
    
    Parameters
    ----------
    data: weka.core.dataset.Instances, weka 형식의 입력 데이터
    asSearch: str, <BestFirst, GreedyStepwise, ...>
    asEvaluator: st, <CfsSubsetEval, ...>
    search_option: list, 탐색하는 방법의 옵션 추가
    asEvaluator: list, 평가하는 방법의 옵션 추가
    verbose: bool, 특징추출의 결과를 상세히 출력
    
    Return
    ----------
    AttributeSelection, 추출한 특징
    
    References
    ----------
    .. [1] M. A. Hall (1998). Correlation-based Feature Subset Selection
           for Machine Learning. Hamilton, New Zealand.  
    .. [2] wikipedia, Feature_selection, <https://en.wikipedia.org/wiki/Feature_selection>  
    .. [3] Weka API Document, CfsSubsetEval,
           <http://weka.sourceforge.net/doc.dev/index.html?weka/attributeSelection/CfsSubsetEval.html>
    """
```
