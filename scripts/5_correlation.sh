fastspar --threads $threads --otu_table $workplace$inputfile --correlation ${workplace}median_correlation.tsv --covariance ${workplace}median_covariance.tsv

mkdir ${workplace}bootstrap_feat
fastspar_bootstrap --otu_table $workplace$inputfile --number 1000 --prefix ${workplace}bootstrap_feat/feat

mkdir ${workplace}$outputfile
parallel fastspar --otu_table {} --correlation $workplace$outputfile/cor_{/} --covariance $workplace$outputfile/cov_{/} -i 5 ::: ${workplace}bootstrap_feat/*

fastspar_pvalues --otu_table $workplace$inputfile --correlation ${workplace}median_correlation.tsv --prefix $workplace$outputfile/cor_ --permutations 1000 --outfile ${workplace}pvalues.tsv
