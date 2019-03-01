import pathlib
from os import path
from weka.core.converters import Loader
from weka.core.converters import Saver
from weka.plot import dataset
from weka.attribute_selection import ASSearch
from weka.attribute_selection import ASEvaluation
from weka.attribute_selection import AttributeSelection



# Hide import in library (Python)
__all__ = [ 'print_title'
           ,'csv2arff'
           ,'load_data' 
           ,'save_train_test_split'
           ,'scatter_plots'
           ,'feature_selection'
          ]

def print_title(title, back_ch='-', width=80, end='\n'):
    """title을 멋지게(?) 프린트해주는 함수
    
    Parameters
    ----------
    title: str, 제목이나 소제목
    back_ch: str, optional, 제목 주위에 표시될 문자
    width: int, optional, 제목까지 합쳐서 표시될 프린트 문자열의 너비
    
    """
    print(back_ch*width)
    print(f' {title} '.center(width, back_ch))
    print(back_ch*width, end=end)
    return


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
    assert(path.exists(fname_csv))
    # 파일명, 확장자 분리
    root, ext = path.splitext(fname_csv)
         
    # arff 파일 생성
    fname_arff = root + '.arff'
    with open(fname_arff, 'w') as fout, open(fname_csv) as fin:
        fout.write(header + '\n')
        list(map(fout.write, iter(fin)))
    return 


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
    # 로드 객체 생성
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(dfile)
    data.class_is_last()  # 예측 클래스를 마지막 Attribute로 설정.
    return data


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
    # 파일경로, 확장자 분리
    root, ext = path.splitext(dfile)
    # 저장할 파일명 구성
    dfile_train = root + '_train' + ext
    dfile_test = root + '_test' + ext
    assert(ext == '.arff')
    
    # weka data 형식을 저장하는 api 클래스
    saver = Saver('weka.core.converters.ArffSaver')
    
    # data 로드, percentage가 70인 경우, train data 70%, test_data 나머지로 분리
    data = load_data(dfile)
    train_data, test_data = data.train_test_split(percentage, rng)
        
    # train, test 데이터 각각 저장
    saver.save_file(train_data, dfile_train)
    saver.save_file(test_data, dfile_test)
    return
    
    
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
    # outpath의 디렉토리가 없는 경우 만든다.
    p = pathlib.Path(outpath)
    p.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    data = load_data(dfile)
    # 특징간의 산점도를 모두 그린다.
    for f1 in data.attributes():
        for f2 in data.attributes():
            dataset.scatter_plot(data,
                             f1.index,
                             f2.index,
                             size=10,
                             outfile='{0}.svg'.format(path.join(outpath, f1.name + ' vs. ' + f2.name)),
                             title=f'{f1.name.capitalize()} vs. {f2.name.capitalize()}')
    

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
    # 탐색과 평가를 담당하는 객체 생성
    search = ASSearch(classname=f"weka.attributeSelection.{asSearch}")
    evaluator = ASEvaluation(classname=f"weka.attributeSelection.{asEvaluator}")
    # 탐색, 평가 객체의 옵션 추가
    if search_option is not None:
        search.options = search_option
    if evaluator_option is not None:
        evaluator.options = evaluator_option
    
    # 특징추출 객체 생성하고, 탐색과 평가 객체를 등록
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    
    # 많은 정보 출력
    if vervose:
        print("# attributes: " + str(attsel.number_attributes_selected))
        print("attributes: " + str(attsel.selected_attributes))
        print("result string:\n" + attsel.results_string)

    return attsel
