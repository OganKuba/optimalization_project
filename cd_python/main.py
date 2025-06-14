import pandas as pd, os
import matplotlib.pyplot as plt
from datasets   import make_dense, make_sparse, california
from experiment import run
from plots      import scatter

os.environ.setdefault("OMP_NUM_THREADS","4")

EXP = [
    #   dataset-factory        lam_start lam_end  η
    (make_dense (500,1000),   0.1, 0.02, 0.85),
    (make_dense (1000,5000),  0.1, 0.02, 0.85),
    (make_sparse(500,1000,0.01), 0.1, 0.02, 0.85),
    (make_sparse(500,10000,0.01),0.1, 0.02, 0.85),
    (california(),            0.1, 0.005,0.80),
]

all_rows=[]
for ds,ls,le,eta in EXP:
    rows=run(ds, lam_start_factor=ls,
                  lam_end_factor=le,
                  eta=eta)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df.to_csv("lasso_cd_bench.csv", index=False, float_format="%.12g")
print("Zapisano lasso_cd_bench.csv")

# --- wykresy: dzielimy gęste / rzadkie po nazwie ds --------
scatter(df[df.ds.str.startswith("dense")],  "Dense – time vs MSE")
scatter(df[df.ds.str.startswith("sparse")], "Sparse – time vs MSE")
scatter(df[df.ds=="california"],            "California – time vs MSE")
plt.show()
