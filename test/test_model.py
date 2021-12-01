import sys
sys.path.insert(1, '../app/')
import model

def test_chiffres():
    assert len(list(model.prediction([1,4,5,8,7,4,2]))) == 2