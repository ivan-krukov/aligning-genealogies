.PHONY: test


fig/coal_example_genealogy.png, fig/coal_example_tree.png: genealogy_aligner/coal_example.py
	python $<


test: test/*
	python -m unitest discover
