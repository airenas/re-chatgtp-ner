log?=INFO
python_cmd=PYTHONPATH=./ LOG_LEVEL=$(log) python
############################################################
files=09_01 09_02 09_03 09_04
work_dir=work
authors=$(patsubst %,$(work_dir)/%_authors.csv,$(files))
authors_json=$(patsubst %,$(work_dir)/%_authors.json,$(files))


$(work_dir):
	mkdir -p $@

.SECONDEXPANSION:
$(work_dir)/%_authors.json: data/orig/item_$$*/item_$$*.in.txt | $(work_dir)
	$(python_cmd) chatgpt_ner/ner_authors.py --input $^ > $@_
	mv $@_ $@
$(work_dir)/%_authors.csv: $(work_dir)/%_authors.json
	$(python_cmd) chatgpt_ner/to_csv.py --input $^ --name $* > $@_
	mv $@_ $@

# $(work_dir)/%_authors.json: data/orig/item_09_04/item_09_04.in.txt | $(work_dir)
# 	python ner-authors.py $^ > $@_
# 	mv $@_ $@

$(work_dir)/out.csv: $(authors)
	$(python_cmd) chatgpt_ner/combine_csv.py $^ > $@_
	mv $@_ $@
    
extract/authors: $(work_dir)/out.csv $(authors_json)

compare/authors: extract/authors
	$(python_cmd) chatgpt_ner/compare.py --gt data/wanted.csv --pred $(work_dir)/out.csv 


$(work_dir)/out-inst.json: data/wanted.csv | $(work_dir)
	$(python_cmd) chatgpt_ner/ner_inst.py --input $^ --cache $@ > $@_
	mv $@_ $@
$(work_dir)/out-inst.csv: $(work_dir)/out-inst.json
	$(python_cmd) chatgpt_ner/inst_to_csv.py --input $^ > $@_
	mv $@_ $@

extract/inst: $(work_dir)/out-inst.csv $(authors_json)

compare/inst: extract/inst
	$(python_cmd) chatgpt_ner/inst_compare.py --gt data/wanted.csv --pred $(work_dir)/out-inst.csv

info:
	echo $(authors)

clean:
	rm -rf $(work_dir)

clean/inst:
	rm -f $(work_dir)/out-inst.json $(work_dir)/out-inst.csv
