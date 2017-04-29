from homework2_rent import main, score_rent


def test_rent():
	"""
	Test model score against assumption.
	"""
	model, X_test, y_test = main()
	r2 = score_rent(model, X_test, y_test)
	
	assert r2 >= .57

if __name__ == '__main__':
	test_rent()
