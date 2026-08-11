.PHONY: pull features train app all

pull:
	python src/pull_data.py

features:
	python src/features.py

train:
	python src/train.py

app:
	streamlit run app.py

all: pull features train
