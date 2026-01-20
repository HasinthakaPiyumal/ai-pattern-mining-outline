# Cluster 6

def grid_search(params):
    single_run(params)
    "\n    for i in range(0,5):      \n        params['dim'] = pow(2,i)    \n        \n        for _ in range(3):\n            single_run(  params)\n    "

