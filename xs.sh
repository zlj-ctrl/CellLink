#!/bin/bash

#SBATCH --job-name='CRC_tumor_1000_0.2_3'
#SBATCH --chdir=/share/home/zhonglaijun/CCC/CRC/KUL28/1000/tumor
#SBATCH --partition=big
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=365-00:00

start_time=$(date +%s.%N)  # 记录开始时间

celltypes=("Epithelial cells" "Stromal cells" "Myeloids" "T cells" "B cells" "Mast cells")
pairs=()
pcc_limit=0.2  # PCC limit value
n_splits=3  # Number of splits

# 生成细胞类型对
for cell1 in "${celltypes[@]}"; do
    for cell2 in "${celltypes[@]}"; do
        pairs+=("$cell1" "$cell2")  # 分别作为两个参数传递
    done
done

# 顺序执行代码
for ((i=0; i<${#pairs[@]}; i+=2)); do
    celltype1=${pairs[i]}
    celltype2=${pairs[i+1]}
    if [ "$celltype1" == "$celltype2" ]; then
        python same.py "$celltype1" "$celltype2" "$pcc_limit" "$n_splits"
    else
        python ccc.py "$celltype1" "$celltype2" "$pcc_limit" "$n_splits"
    fi
done

end_time=$(date +%s.%N)  # 记录结束时间
runtime=$(echo "$end_time - $start_time" | bc)  # 计算运行时间
echo "Total runtime: $runtime seconds"
