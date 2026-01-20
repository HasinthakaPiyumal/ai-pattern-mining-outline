# Cluster 65

class TestLarge_int_format(TestCase):

    def test_large_int_format(self):
        self.assertEqual(large_int_format(1), '1')
        self.assertEqual(large_int_format(6), '6')
        self.assertEqual(large_int_format(10), '10')
        self.assertEqual(large_int_format(19), '10')
        self.assertEqual(large_int_format(88), '80')
        self.assertEqual(large_int_format(999), '900')
        self.assertEqual(large_int_format(1001), '1k')
        self.assertEqual(large_int_format(205001), '200k')
        self.assertEqual(large_int_format(2050010), '2mm')
        self.assertEqual(large_int_format(205000010), '200mm')
        self.assertEqual(large_int_format(2050000010), '2b')

def large_int_format(x):
    num = round_downer(x)
    if 1000000000 <= num:
        return str(num // 1000000000) + 'b'
    elif 1000000 <= num < 1000000000:
        return str(num // 1000000) + 'mm'
    elif 1000 <= num < 1000000:
        return str(num // 1000) + 'k'
    else:
        return str(num)

