#import pysal as ps
from libpysal.io import geotable
from libpysal import examples, weights
import numpy as np
from sklearn.metrics import pairwise as skm
from region.skater.skater import Spanning_Forest

import types
import os
TESTDIR = os.path.dirname(os.path.abspath(__file__))


#df = ps.pdio.read_files(ps.examples.get_path('south.shp'))
df = geotable.read_files(examples.get_path('south.shp'))
data = df[df.filter(like='90').columns.tolist()
               + df.filter(like='89').columns.tolist()].values
data_c = (data - data.mean(axis=0)) / data.std(axis=0)
W = weights.Queen.from_dataframe(df)
    
def test_init():
    default = Spanning_Forest()
    assert default.metric == skm.manhattan_distances
    assert default.center == np.mean
    assert default.reduction == np.sum
    change = Spanning_Forest(dissimilarity=skm.euclidean_distances,
                             center=np.median, reduction=np.max)
    assert change.metric == skm.euclidean_distances
    assert change.center == np.median
    assert change.reduction == np.max

    sym = Spanning_Forest(affinity=skm.cosine_similarity)
    assert isinstance(sym.metric, types.LambdaType)
    test_distance = -np.log(skm.cosine_similarity(data[:2,]))
    comparator = sym.metric(data[:2,])
    np.testing.assert_allclose(test_distance, comparator)

def test_run():
    result = Spanning_Forest().fit(5, W, data_c, quorum=50)

    #hmm... numbering is arbitrary... just check that they run for now
    #np.testing.assert_array_equal(self.south_5_q100, result.current_labels_)
    result2 = Spanning_Forest().fit(np.inf, W, data_c, quorum=20)

    #np.testing.assert_array_equal(self.south_inf_q20, result2.current_labels_)

