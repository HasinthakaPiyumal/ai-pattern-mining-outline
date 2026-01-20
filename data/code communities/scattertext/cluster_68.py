# Cluster 68

class TestPercentile_lexicographic(TestCase):

    def test_percentile_lexicographic(self):
        scores = [1, 1, 5, 18, 1, 3]
        text = ['c', 'a', 'five', 'eighteen', 'b', 'three']
        ranking = percentile_alphabetical(scores, text)
        np.testing.assert_array_almost_equal(ranking, np.array([0.4, 0, 0.8, 1.0, 0.2, 0.6]))

def percentile_alphabetical(vec, terms, other_vec=None):
    scale_df = pd.DataFrame({'scores': vec, 'terms': terms})
    if other_vec is not None:
        scale_df['others'] = other_vec
    else:
        scale_df['others'] = 0
    vec_ss = scale_df.sort_values(by=['scores', 'terms', 'others'], ascending=[True, True, False]).reset_index().sort_values(by='index').index
    return scale_0_to_1(vec_ss)

