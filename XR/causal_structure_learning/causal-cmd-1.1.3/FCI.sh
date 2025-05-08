java -jar causal-cmd-1.1.3-jar-with-dependencies.jar --dataset $1 --data-type discrete --algorithm fci --delimiter comma --alpha $2 \
--out $3 --depth $4 --completeRuleSetUsed --test chi-square-test --knowledge knowledge.txt
