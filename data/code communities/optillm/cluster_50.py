# Cluster 50

def run_imo25_tests():
    """Run all IMO25 MARS tests"""
    print('Running MARS IMO25 specific tests...')
    print('=' * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMARSIMO25)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print('=' * 80)
    if result.wasSuccessful():
        print('🎉 All IMO25 tests passed!')
        return True
    else:
        print('❌ Some IMO25 tests failed - analyzing for improvements')
        return False

