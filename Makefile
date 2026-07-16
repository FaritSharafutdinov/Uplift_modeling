.PHONY: all clean cohort matching aipw cate report

all: cohort matching aipw cate report

cohort:
	python 01_cohort_creation.py
	python cohort_counts.py

matching:
	python 02_propensity_matching.py

aipw:
	python 03_aipw_analysis.py

cate:
	python 04_cate_analysis.py

report:
	python 05_summary_results.py
	python 06_generate_report.py

clean:
	rm -f *.pkl *.png *.csv *.json *.log
	rm -f FINAL_REPORT.md
