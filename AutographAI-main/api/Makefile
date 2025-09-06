UNIT_TEST_PATH=components
PYTHONPATH=$(shell pwd)/src
TEST_PATTERN=*unit_test.py

unit-test-components:
	@echo "Python path is : $(PYTHONPATH)"
	@echo "Running unit tests in $(UNIT_TEST_PATH) with PYTHONPATH=$(PYTHONPATH)"
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s $(UNIT_TEST_PATH) -p "$(TEST_PATTERN)"
