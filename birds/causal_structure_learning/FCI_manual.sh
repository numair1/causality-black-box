java -jar causal-cmd-1.3.0-jar-with-dependencies.jar --dataset ./bootdata_post_FACCT/subsample_0.csv --data-type discrete --algorithm fci --delimiter tab --alpha 0.00001 \
--out ./ --depth 5 --completeRuleSetUsed --test chi-square-test --knowledge knowledge.txt --verbose --maxPathLength 10 --stableFAS --out FCI_birds_subsample0_1e5_depth5.txt 
