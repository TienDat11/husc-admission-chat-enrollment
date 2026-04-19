/content/husc-admission-chat-enrollment/rag2025
Using BASE_PATH: /content/husc-admission-chat-enrollment/rag2025
total 372
drwxr-xr-x 2 root root  4096 Apr 18 17:21 .
drwxr-xr-x 8 root root  4096 Apr 18 17:21 ..
-rw-r--r-- 1 root root 60054 Apr 18 17:19 chunked_10_enhanced.jsonl
-rw-r--r-- 1 root root 52830 Apr 18 17:19 chunked_10.jsonl
-rw-r--r-- 1 root root 20083 Apr 18 17:19 chunked_11.jsonl
-rw-r--r-- 1 root root 43984 Apr 18 17:19 chunked_1.jsonl
-rw-r--r-- 1 root root  3933 Apr 18 17:19 chunked_2.jsonl
-rw-r--r-- 1 root root 87110 Apr 18 17:19 chunked_3.jsonl
-rw-r--r-- 1 root root  4980 Apr 18 17:19 chunked_4.jsonl
-rw-r--r-- 1 root root 21710 Apr 18 17:19 chunked_5.jsonl
-rw-r--r-- 1 root root  6070 Apr 18 17:19 chunked_6.jsonl
-rw-r--r-- 1 root root     0 Apr 18 17:19 chunked_7.jsonl
-rw-r--r-- 1 root root 34526 Apr 18 17:19 chunked_8.jsonl
-rw-r--r-- 1 root root  8400 Apr 18 17:19 chunked_9.jsonl
-rw-r--r-- 1 root root  3564 Apr 18 17:19 chunked_test.jsonl
-rw-r--r-- 1 root root  1075 Apr 18 17:21 .ingest_manifest.json
---

>>> Step 1: LanceDB Ingest
2026-04-18 18:03:08.033 | INFO     | __main__:ingest_lancedb:140 - Loading chunks from 12 JSONL file(s)...
2026-04-18 18:03:08.038 | INFO     | __main__:ingest_lancedb:150 - Loaded 152 chunks
2026-04-18 18:03:08.038 | INFO     | __main__:ingest_lancedb:151 - Loading embedding model: BAAI/bge-m3 (dim=1024)
2026-04-18 18:03:08.038 | INFO     | src.services.embedding:__init__:30 - Loading embedding model: BAAI/bge-m3 (provider=bge, dim=1024)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/391 [00:00<?, ?it/s]
Loading weights:   0%|          | 1/391 [00:00<00:00, 10381.94it/s, Materializing param=embeddings.LayerNorm.bias]
Loading weights:   0%|          | 1/391 [00:00<00:00, 2991.66it/s, Materializing param=embeddings.LayerNorm.bias] 
Loading weights:   1%|          | 2/391 [00:00<00:00, 3315.66it/s, Materializing param=embeddings.LayerNorm.weight]
Loading weights:   1%|          | 2/391 [00:00<00:00, 2753.07it/s, Materializing param=embeddings.LayerNorm.weight]
Loading weights:   1%|          | 3/391 [00:00<00:00, 3199.32it/s, Materializing param=embeddings.position_embeddings.weight]
Loading weights:   1%|          | 3/391 [00:00<00:00, 2803.05it/s, Materializing param=embeddings.position_embeddings.weight]
Loading weights:   1%|          | 4/391 [00:00<00:00, 3191.40it/s, Materializing param=embeddings.token_type_embeddings.weight]
Loading weights:   1%|          | 4/391 [00:00<00:00, 2909.68it/s, Materializing param=embeddings.token_type_embeddings.weight]
Loading weights:   1%|▏         | 5/391 [00:00<00:00, 3259.99it/s, Materializing param=embeddings.word_embeddings.weight]      
Loading weights:   1%|▏         | 5/391 [00:00<00:00, 3045.09it/s, Materializing param=embeddings.word_embeddings.weight]
Loading weights:   2%|▏         | 6/391 [00:00<00:00, 3332.34it/s, Materializing param=encoder.layer.0.attention.output.LayerNorm.bias]
Loading weights:   2%|▏         | 6/391 [00:00<00:00, 3107.28it/s, Materializing param=encoder.layer.0.attention.output.LayerNorm.bias]
Loading weights:   2%|▏         | 7/391 [00:00<00:00, 3221.08it/s, Materializing param=encoder.layer.0.attention.output.LayerNorm.weight]
Loading weights:   2%|▏         | 7/391 [00:00<00:00, 3034.95it/s, Materializing param=encoder.layer.0.attention.output.LayerNorm.weight]
Loading weights:   2%|▏         | 8/391 [00:00<00:00, 3188.07it/s, Materializing param=encoder.layer.0.attention.output.dense.bias]      
Loading weights:   2%|▏         | 8/391 [00:00<00:00, 3038.53it/s, Materializing param=encoder.layer.0.attention.output.dense.bias]
Loading weights:   2%|▏         | 9/391 [00:00<00:00, 3178.57it/s, Materializing param=encoder.layer.0.attention.output.dense.weight]
Loading weights:   2%|▏         | 9/391 [00:00<00:00, 3010.75it/s, Materializing param=encoder.layer.0.attention.output.dense.weight]
Loading weights:   3%|▎         | 10/391 [00:00<00:00, 3120.99it/s, Materializing param=encoder.layer.0.attention.self.key.bias]     
Loading weights:   3%|▎         | 10/391 [00:00<00:00, 3006.24it/s, Materializing param=encoder.layer.0.attention.self.key.bias]
Loading weights:   3%|▎         | 11/391 [00:00<00:00, 3122.66it/s, Materializing param=encoder.layer.0.attention.self.key.weight]
Loading weights:   3%|▎         | 11/391 [00:00<00:00, 3020.25it/s, Materializing param=encoder.layer.0.attention.self.key.weight]
Loading weights:   3%|▎         | 12/391 [00:00<00:00, 3130.27it/s, Materializing param=encoder.layer.0.attention.self.query.bias]
Loading weights:   3%|▎         | 12/391 [00:00<00:00, 3021.47it/s, Materializing param=encoder.layer.0.attention.self.query.bias]
Loading weights:   3%|▎         | 13/391 [00:00<00:00, 3113.10it/s, Materializing param=encoder.layer.0.attention.self.query.weight]
Loading weights:   3%|▎         | 13/391 [00:00<00:00, 3023.51it/s, Materializing param=encoder.layer.0.attention.self.query.weight]
Loading weights:   4%|▎         | 14/391 [00:00<00:00, 3112.66it/s, Materializing param=encoder.layer.0.attention.self.value.bias]  
Loading weights:   4%|▎         | 14/391 [00:00<00:00, 3029.78it/s, Materializing param=encoder.layer.0.attention.self.value.bias]
Loading weights:   4%|▍         | 15/391 [00:00<00:00, 3116.74it/s, Materializing param=encoder.layer.0.attention.self.value.weight]
Loading weights:   4%|▍         | 15/391 [00:00<00:00, 3029.25it/s, Materializing param=encoder.layer.0.attention.self.value.weight]
Loading weights:   4%|▍         | 16/391 [00:00<00:00, 3100.72it/s, Materializing param=encoder.layer.0.intermediate.dense.bias]    
Loading weights:   4%|▍         | 16/391 [00:00<00:00, 3027.70it/s, Materializing param=encoder.layer.0.intermediate.dense.bias]
Loading weights:   4%|▍         | 17/391 [00:00<00:00, 3103.92it/s, Materializing param=encoder.layer.0.intermediate.dense.weight]
Loading weights:   4%|▍         | 17/391 [00:00<00:00, 3037.41it/s, Materializing param=encoder.layer.0.intermediate.dense.weight]
Loading weights:   5%|▍         | 18/391 [00:00<00:00, 3113.04it/s, Materializing param=encoder.layer.0.output.LayerNorm.bias]    
Loading weights:   5%|▍         | 18/391 [00:00<00:00, 3042.54it/s, Materializing param=encoder.layer.0.output.LayerNorm.bias]
Loading weights:   5%|▍         | 19/391 [00:00<00:00, 3105.20it/s, Materializing param=encoder.layer.0.output.LayerNorm.weight]
Loading weights:   5%|▍         | 19/391 [00:00<00:00, 3050.40it/s, Materializing param=encoder.layer.0.output.LayerNorm.weight]
Loading weights:   5%|▌         | 20/391 [00:00<00:00, 3127.28it/s, Materializing param=encoder.layer.0.output.dense.bias]      
Loading weights:   5%|▌         | 20/391 [00:00<00:00, 3077.48it/s, Materializing param=encoder.layer.0.output.dense.bias]
Loading weights:   5%|▌         | 21/391 [00:00<00:00, 3150.12it/s, Materializing param=encoder.layer.0.output.dense.weight]
Loading weights:   5%|▌         | 21/391 [00:00<00:00, 3101.20it/s, Materializing param=encoder.layer.0.output.dense.weight]
Loading weights:   6%|▌         | 22/391 [00:00<00:00, 3138.59it/s, Materializing param=encoder.layer.1.attention.output.LayerNorm.bias]
Loading weights:   6%|▌         | 22/391 [00:00<00:00, 3075.41it/s, Materializing param=encoder.layer.1.attention.output.LayerNorm.bias]
Loading weights:   6%|▌         | 23/391 [00:00<00:00, 3123.29it/s, Materializing param=encoder.layer.1.attention.output.LayerNorm.weight]
Loading weights:   6%|▌         | 23/391 [00:00<00:00, 3069.04it/s, Materializing param=encoder.layer.1.attention.output.LayerNorm.weight]
Loading weights:   6%|▌         | 24/391 [00:00<00:00, 3117.86it/s, Materializing param=encoder.layer.1.attention.output.dense.bias]      
Loading weights:   6%|▌         | 24/391 [00:00<00:00, 3052.72it/s, Materializing param=encoder.layer.1.attention.output.dense.bias]
Loading weights:   6%|▋         | 25/391 [00:00<00:00, 3092.87it/s, Materializing param=encoder.layer.1.attention.output.dense.weight]
Loading weights:   6%|▋         | 25/391 [00:00<00:00, 3041.20it/s, Materializing param=encoder.layer.1.attention.output.dense.weight]
Loading weights:   7%|▋         | 26/391 [00:00<00:00, 3090.25it/s, Materializing param=encoder.layer.1.attention.self.key.bias]      
Loading weights:   7%|▋         | 26/391 [00:00<00:00, 3046.74it/s, Materializing param=encoder.layer.1.attention.self.key.bias]
Loading weights:   7%|▋         | 27/391 [00:00<00:00, 3086.57it/s, Materializing param=encoder.layer.1.attention.self.key.weight]
Loading weights:   7%|▋         | 27/391 [00:00<00:00, 3040.33it/s, Materializing param=encoder.layer.1.attention.self.key.weight]
Loading weights:   7%|▋         | 28/391 [00:00<00:00, 3076.61it/s, Materializing param=encoder.layer.1.attention.self.query.bias]
Loading weights:   7%|▋         | 28/391 [00:00<00:00, 3039.51it/s, Materializing param=encoder.layer.1.attention.self.query.bias]
Loading weights:   7%|▋         | 29/391 [00:00<00:00, 3090.32it/s, Materializing param=encoder.layer.1.attention.self.query.weight]
Loading weights:   7%|▋         | 29/391 [00:00<00:00, 3054.85it/s, Materializing param=encoder.layer.1.attention.self.query.weight]
Loading weights:   8%|▊         | 30/391 [00:00<00:00, 3104.82it/s, Materializing param=encoder.layer.1.attention.self.value.bias]  
Loading weights:   8%|▊         | 30/391 [00:00<00:00, 3071.10it/s, Materializing param=encoder.layer.1.attention.self.value.bias]
Loading weights:   8%|▊         | 31/391 [00:00<00:00, 3112.84it/s, Materializing param=encoder.layer.1.attention.self.value.weight]
Loading weights:   8%|▊         | 31/391 [00:00<00:00, 3076.53it/s, Materializing param=encoder.layer.1.attention.self.value.weight]
Loading weights:   8%|▊         | 32/391 [00:00<00:00, 3121.85it/s, Materializing param=encoder.layer.1.intermediate.dense.bias]    
Loading weights:   8%|▊         | 32/391 [00:00<00:00, 3090.30it/s, Materializing param=encoder.layer.1.intermediate.dense.bias]
Loading weights:   8%|▊         | 33/391 [00:00<00:00, 3137.53it/s, Materializing param=encoder.layer.1.intermediate.dense.weight]
Loading weights:   8%|▊         | 33/391 [00:00<00:00, 3106.12it/s, Materializing param=encoder.layer.1.intermediate.dense.weight]
Loading weights:   9%|▊         | 34/391 [00:00<00:00, 3151.94it/s, Materializing param=encoder.layer.1.output.LayerNorm.bias]    
Loading weights:   9%|▊         | 34/391 [00:00<00:00, 3117.08it/s, Materializing param=encoder.layer.1.output.LayerNorm.bias]
Loading weights:   9%|▉         | 35/391 [00:00<00:00, 3148.94it/s, Materializing param=encoder.layer.1.output.LayerNorm.weight]
Loading weights:   9%|▉         | 35/391 [00:00<00:00, 3114.74it/s, Materializing param=encoder.layer.1.output.LayerNorm.weight]
Loading weights:   9%|▉         | 36/391 [00:00<00:00, 3149.53it/s, Materializing param=encoder.layer.1.output.dense.bias]      
Loading weights:   9%|▉         | 36/391 [00:00<00:00, 3117.03it/s, Materializing param=encoder.layer.1.output.dense.bias]
Loading weights:   9%|▉         | 37/391 [00:00<00:00, 3153.04it/s, Materializing param=encoder.layer.1.output.dense.weight]
Loading weights:   9%|▉         | 37/391 [00:00<00:00, 3117.94it/s, Materializing param=encoder.layer.1.output.dense.weight]
Loading weights:  10%|▉         | 38/391 [00:00<00:00, 3150.74it/s, Materializing param=encoder.layer.2.attention.output.LayerNorm.bias]
Loading weights:  10%|▉         | 38/391 [00:00<00:00, 3117.34it/s, Materializing param=encoder.layer.2.attention.output.LayerNorm.bias]
Loading weights:  10%|▉         | 39/391 [00:00<00:00, 3148.51it/s, Materializing param=encoder.layer.2.attention.output.LayerNorm.weight]
Loading weights:  10%|▉         | 39/391 [00:00<00:00, 3117.67it/s, Materializing param=encoder.layer.2.attention.output.LayerNorm.weight]
Loading weights:  10%|█         | 40/391 [00:00<00:00, 3149.17it/s, Materializing param=encoder.layer.2.attention.output.dense.bias]      
Loading weights:  10%|█         | 40/391 [00:00<00:00, 3119.72it/s, Materializing param=encoder.layer.2.attention.output.dense.bias]
Loading weights:  10%|█         | 41/391 [00:00<00:00, 3145.77it/s, Materializing param=encoder.layer.2.attention.output.dense.weight]
Loading weights:  10%|█         | 41/391 [00:00<00:00, 3114.88it/s, Materializing param=encoder.layer.2.attention.output.dense.weight]
Loading weights:  11%|█         | 42/391 [00:00<00:00, 3145.17it/s, Materializing param=encoder.layer.2.attention.self.key.bias]      
Loading weights:  11%|█         | 42/391 [00:00<00:00, 3117.39it/s, Materializing param=encoder.layer.2.attention.self.key.bias]
Loading weights:  11%|█         | 43/391 [00:00<00:00, 3147.06it/s, Materializing param=encoder.layer.2.attention.self.key.weight]
Loading weights:  11%|█         | 43/391 [00:00<00:00, 3120.22it/s, Materializing param=encoder.layer.2.attention.self.key.weight]
Loading weights:  11%|█▏        | 44/391 [00:00<00:00, 3145.66it/s, Materializing param=encoder.layer.2.attention.self.query.bias]
Loading weights:  11%|█▏        | 44/391 [00:00<00:00, 3117.44it/s, Materializing param=encoder.layer.2.attention.self.query.bias]
Loading weights:  12%|█▏        | 45/391 [00:00<00:00, 3145.62it/s, Materializing param=encoder.layer.2.attention.self.query.weight]
Loading weights:  12%|█▏        | 45/391 [00:00<00:00, 3119.58it/s, Materializing param=encoder.layer.2.attention.self.query.weight]
Loading weights:  12%|█▏        | 46/391 [00:00<00:00, 3148.31it/s, Materializing param=encoder.layer.2.attention.self.value.bias]  
Loading weights:  12%|█▏        | 46/391 [00:00<00:00, 3123.19it/s, Materializing param=encoder.layer.2.attention.self.value.bias]
Loading weights:  12%|█▏        | 47/391 [00:00<00:00, 3147.22it/s, Materializing param=encoder.layer.2.attention.self.value.weight]
Loading weights:  12%|█▏        | 47/391 [00:00<00:00, 3121.45it/s, Materializing param=encoder.layer.2.attention.self.value.weight]
Loading weights:  12%|█▏        | 48/391 [00:00<00:00, 3146.07it/s, Materializing param=encoder.layer.2.intermediate.dense.bias]    
Loading weights:  12%|█▏        | 48/391 [00:00<00:00, 3121.73it/s, Materializing param=encoder.layer.2.intermediate.dense.bias]
Loading weights:  13%|█▎        | 49/391 [00:00<00:00, 3148.49it/s, Materializing param=encoder.layer.2.intermediate.dense.weight]
Loading weights:  13%|█▎        | 49/391 [00:00<00:00, 3124.65it/s, Materializing param=encoder.layer.2.intermediate.dense.weight]
Loading weights:  13%|█▎        | 50/391 [00:00<00:00, 3149.40it/s, Materializing param=encoder.layer.2.output.LayerNorm.bias]    
Loading weights:  13%|█▎        | 50/391 [00:00<00:00, 3127.14it/s, Materializing param=encoder.layer.2.output.LayerNorm.bias]
Loading weights:  13%|█▎        | 51/391 [00:00<00:00, 3152.96it/s, Materializing param=encoder.layer.2.output.LayerNorm.weight]
Loading weights:  13%|█▎        | 51/391 [00:00<00:00, 3131.96it/s, Materializing param=encoder.layer.2.output.LayerNorm.weight]
Loading weights:  13%|█▎        | 52/391 [00:00<00:00, 3161.38it/s, Materializing param=encoder.layer.2.output.dense.bias]      
Loading weights:  13%|█▎        | 52/391 [00:00<00:00, 3141.66it/s, Materializing param=encoder.layer.2.output.dense.bias]
Loading weights:  14%|█▎        | 53/391 [00:00<00:00, 3170.80it/s, Materializing param=encoder.layer.2.output.dense.weight]
Loading weights:  14%|█▎        | 53/391 [00:00<00:00, 3150.97it/s, Materializing param=encoder.layer.2.output.dense.weight]
Loading weights:  14%|█▍        | 54/391 [00:00<00:00, 3175.10it/s, Materializing param=encoder.layer.3.attention.output.LayerNorm.bias]
Loading weights:  14%|█▍        | 54/391 [00:00<00:00, 3153.48it/s, Materializing param=encoder.layer.3.attention.output.LayerNorm.bias]
Loading weights:  14%|█▍        | 55/391 [00:00<00:00, 3178.90it/s, Materializing param=encoder.layer.3.attention.output.LayerNorm.weight]
Loading weights:  14%|█▍        | 55/391 [00:00<00:00, 3159.14it/s, Materializing param=encoder.layer.3.attention.output.LayerNorm.weight]
Loading weights:  14%|█▍        | 56/391 [00:00<00:00, 3185.52it/s, Materializing param=encoder.layer.3.attention.output.dense.bias]      
Loading weights:  14%|█▍        | 56/391 [00:00<00:00, 3166.41it/s, Materializing param=encoder.layer.3.attention.output.dense.bias]
Loading weights:  15%|█▍        | 57/391 [00:00<00:00, 3190.22it/s, Materializing param=encoder.layer.3.attention.output.dense.weight]
Loading weights:  15%|█▍        | 57/391 [00:00<00:00, 3167.44it/s, Materializing param=encoder.layer.3.attention.output.dense.weight]
Loading weights:  15%|█▍        | 58/391 [00:00<00:00, 3186.58it/s, Materializing param=encoder.layer.3.attention.self.key.bias]      
Loading weights:  15%|█▍        | 58/391 [00:00<00:00, 3165.39it/s, Materializing param=encoder.layer.3.attention.self.key.bias]
Loading weights:  15%|█▌        | 59/391 [00:00<00:00, 3180.36it/s, Materializing param=encoder.layer.3.attention.self.key.weight]
Loading weights:  15%|█▌        | 59/391 [00:00<00:00, 3160.06it/s, Materializing param=encoder.layer.3.attention.self.key.weight]
Loading weights:  15%|█▌        | 60/391 [00:00<00:00, 3169.98it/s, Materializing param=encoder.layer.3.attention.self.query.bias]
Loading weights:  15%|█▌        | 60/391 [00:00<00:00, 3145.85it/s, Materializing param=encoder.layer.3.attention.self.query.bias]
Loading weights:  16%|█▌        | 61/391 [00:00<00:00, 3162.93it/s, Materializing param=encoder.layer.3.attention.self.query.weight]
Loading weights:  16%|█▌        | 61/391 [00:00<00:00, 3141.34it/s, Materializing param=encoder.layer.3.attention.self.query.weight]
Loading weights:  16%|█▌        | 62/391 [00:00<00:00, 3158.17it/s, Materializing param=encoder.layer.3.attention.self.value.bias]  
Loading weights:  16%|█▌        | 62/391 [00:00<00:00, 3138.84it/s, Materializing param=encoder.layer.3.attention.self.value.bias]
Loading weights:  16%|█▌        | 63/391 [00:00<00:00, 3156.14it/s, Materializing param=encoder.layer.3.attention.self.value.weight]
Loading weights:  16%|█▌        | 63/391 [00:00<00:00, 3135.39it/s, Materializing param=encoder.layer.3.attention.self.value.weight]
Loading weights:  16%|█▋        | 64/391 [00:00<00:00, 3154.46it/s, Materializing param=encoder.layer.3.intermediate.dense.bias]    
Loading weights:  16%|█▋        | 64/391 [00:00<00:00, 3135.85it/s, Materializing param=encoder.layer.3.intermediate.dense.bias]
Loading weights:  17%|█▋        | 65/391 [00:00<00:00, 3155.91it/s, Materializing param=encoder.layer.3.intermediate.dense.weight]
Loading weights:  17%|█▋        | 65/391 [00:00<00:00, 3137.82it/s, Materializing param=encoder.layer.3.intermediate.dense.weight]
Loading weights:  17%|█▋        | 66/391 [00:00<00:00, 3154.76it/s, Materializing param=encoder.layer.3.output.LayerNorm.bias]    
Loading weights:  17%|█▋        | 66/391 [00:00<00:00, 3135.96it/s, Materializing param=encoder.layer.3.output.LayerNorm.bias]
Loading weights:  17%|█▋        | 67/391 [00:00<00:00, 3145.85it/s, Materializing param=encoder.layer.3.output.LayerNorm.weight]
Loading weights:  17%|█▋        | 67/391 [00:00<00:00, 3124.86it/s, Materializing param=encoder.layer.3.output.LayerNorm.weight]
Loading weights:  17%|█▋        | 68/391 [00:00<00:00, 3138.86it/s, Materializing param=encoder.layer.3.output.dense.bias]      
Loading weights:  17%|█▋        | 68/391 [00:00<00:00, 3118.65it/s, Materializing param=encoder.layer.3.output.dense.bias]
Loading weights:  18%|█▊        | 69/391 [00:00<00:00, 3129.94it/s, Materializing param=encoder.layer.3.output.dense.weight]
Loading weights:  18%|█▊        | 69/391 [00:00<00:00, 3108.46it/s, Materializing param=encoder.layer.3.output.dense.weight]
Loading weights:  18%|█▊        | 70/391 [00:00<00:00, 3120.73it/s, Materializing param=encoder.layer.4.attention.output.LayerNorm.bias]
Loading weights:  18%|█▊        | 70/391 [00:00<00:00, 3103.48it/s, Materializing param=encoder.layer.4.attention.output.LayerNorm.bias]
Loading weights:  18%|█▊        | 71/391 [00:00<00:00, 3120.96it/s, Materializing param=encoder.layer.4.attention.output.LayerNorm.weight]
Loading weights:  18%|█▊        | 71/391 [00:00<00:00, 3102.94it/s, Materializing param=encoder.layer.4.attention.output.LayerNorm.weight]
Loading weights:  18%|█▊        | 72/391 [00:00<00:00, 3117.89it/s, Materializing param=encoder.layer.4.attention.output.dense.bias]      
Loading weights:  18%|█▊        | 72/391 [00:00<00:00, 3101.53it/s, Materializing param=encoder.layer.4.attention.output.dense.bias]
Loading weights:  19%|█▊        | 73/391 [00:00<00:00, 3119.05it/s, Materializing param=encoder.layer.4.attention.output.dense.weight]
Loading weights:  19%|█▊        | 73/391 [00:00<00:00, 3104.56it/s, Materializing param=encoder.layer.4.attention.output.dense.weight]
Loading weights:  19%|█▉        | 74/391 [00:00<00:00, 3124.44it/s, Materializing param=encoder.layer.4.attention.self.key.bias]      
Loading weights:  19%|█▉        | 74/391 [00:00<00:00, 3110.38it/s, Materializing param=encoder.layer.4.attention.self.key.bias]
Loading weights:  19%|█▉        | 75/391 [00:00<00:00, 3127.15it/s, Materializing param=encoder.layer.4.attention.self.key.weight]
Loading weights:  19%|█▉        | 75/391 [00:00<00:00, 3112.09it/s, Materializing param=encoder.layer.4.attention.self.key.weight]
Loading weights:  19%|█▉        | 76/391 [00:00<00:00, 3131.74it/s, Materializing param=encoder.layer.4.attention.self.query.bias]
Loading weights:  19%|█▉        | 76/391 [00:00<00:00, 3118.47it/s, Materializing param=encoder.layer.4.attention.self.query.bias]
Loading weights:  20%|█▉        | 77/391 [00:00<00:00, 3138.26it/s, Materializing param=encoder.layer.4.attention.self.query.weight]
Loading weights:  20%|█▉        | 77/391 [00:00<00:00, 3125.17it/s, Materializing param=encoder.layer.4.attention.self.query.weight]
Loading weights:  20%|█▉        | 78/391 [00:00<00:00, 3145.03it/s, Materializing param=encoder.layer.4.attention.self.value.bias]  
Loading weights:  20%|█▉        | 78/391 [00:00<00:00, 3128.88it/s, Materializing param=encoder.layer.4.attention.self.value.bias]
Loading weights:  20%|██        | 79/391 [00:00<00:00, 3142.66it/s, Materializing param=encoder.layer.4.attention.self.value.weight]
Loading weights:  20%|██        | 79/391 [00:00<00:00, 3127.21it/s, Materializing param=encoder.layer.4.attention.self.value.weight]
Loading weights:  20%|██        | 80/391 [00:00<00:00, 3142.62it/s, Materializing param=encoder.layer.4.intermediate.dense.bias]    
Loading weights:  20%|██        | 80/391 [00:00<00:00, 3127.89it/s, Materializing param=encoder.layer.4.intermediate.dense.bias]
Loading weights:  21%|██        | 81/391 [00:00<00:00, 3142.79it/s, Materializing param=encoder.layer.4.intermediate.dense.weight]
Loading weights:  21%|██        | 81/391 [00:00<00:00, 3126.33it/s, Materializing param=encoder.layer.4.intermediate.dense.weight]
Loading weights:  21%|██        | 82/391 [00:00<00:00, 3139.45it/s, Materializing param=encoder.layer.4.output.LayerNorm.bias]    
Loading weights:  21%|██        | 82/391 [00:00<00:00, 3124.90it/s, Materializing param=encoder.layer.4.output.LayerNorm.bias]
Loading weights:  21%|██        | 83/391 [00:00<00:00, 3139.93it/s, Materializing param=encoder.layer.4.output.LayerNorm.weight]
Loading weights:  21%|██        | 83/391 [00:00<00:00, 3125.75it/s, Materializing param=encoder.layer.4.output.LayerNorm.weight]
Loading weights:  21%|██▏       | 84/391 [00:00<00:00, 3141.44it/s, Materializing param=encoder.layer.4.output.dense.bias]      
Loading weights:  21%|██▏       | 84/391 [00:00<00:00, 3126.24it/s, Materializing param=encoder.layer.4.output.dense.bias]
Loading weights:  22%|██▏       | 85/391 [00:00<00:00, 3139.75it/s, Materializing param=encoder.layer.4.output.dense.weight]
Loading weights:  22%|██▏       | 85/391 [00:00<00:00, 3125.96it/s, Materializing param=encoder.layer.4.output.dense.weight]
Loading weights:  22%|██▏       | 86/391 [00:00<00:00, 3140.95it/s, Materializing param=encoder.layer.5.attention.output.LayerNorm.bias]
Loading weights:  22%|██▏       | 86/391 [00:00<00:00, 3127.20it/s, Materializing param=encoder.layer.5.attention.output.LayerNorm.bias]
Loading weights:  22%|██▏       | 87/391 [00:00<00:00, 3141.34it/s, Materializing param=encoder.layer.5.attention.output.LayerNorm.weight]
Loading weights:  22%|██▏       | 87/391 [00:00<00:00, 3127.61it/s, Materializing param=encoder.layer.5.attention.output.LayerNorm.weight]
Loading weights:  23%|██▎       | 88/391 [00:00<00:00, 3139.42it/s, Materializing param=encoder.layer.5.attention.output.dense.bias]      
Loading weights:  23%|██▎       | 88/391 [00:00<00:00, 3124.83it/s, Materializing param=encoder.layer.5.attention.output.dense.bias]
Loading weights:  23%|██▎       | 89/391 [00:00<00:00, 3138.39it/s, Materializing param=encoder.layer.5.attention.output.dense.weight]
Loading weights:  23%|██▎       | 89/391 [00:00<00:00, 3124.94it/s, Materializing param=encoder.layer.5.attention.output.dense.weight]
Loading weights:  23%|██▎       | 90/391 [00:00<00:00, 3138.64it/s, Materializing param=encoder.layer.5.attention.self.key.bias]      
Loading weights:  23%|██▎       | 90/391 [00:00<00:00, 3125.34it/s, Materializing param=encoder.layer.5.attention.self.key.bias]
Loading weights:  23%|██▎       | 91/391 [00:00<00:00, 3136.33it/s, Materializing param=encoder.layer.5.attention.self.key.weight]
Loading weights:  23%|██▎       | 91/391 [00:00<00:00, 3122.19it/s, Materializing param=encoder.layer.5.attention.self.key.weight]
Loading weights:  24%|██▎       | 92/391 [00:00<00:00, 3136.08it/s, Materializing param=encoder.layer.5.attention.self.query.bias]
Loading weights:  24%|██▎       | 92/391 [00:00<00:00, 3124.45it/s, Materializing param=encoder.layer.5.attention.self.query.bias]
Loading weights:  24%|██▍       | 93/391 [00:00<00:00, 3140.16it/s, Materializing param=encoder.layer.5.attention.self.query.weight]
Loading weights:  24%|██▍       | 93/391 [00:00<00:00, 3128.72it/s, Materializing param=encoder.layer.5.attention.self.query.weight]
Loading weights:  24%|██▍       | 94/391 [00:00<00:00, 3140.80it/s, Materializing param=encoder.layer.5.attention.self.value.bias]  
Loading weights:  24%|██▍       | 94/391 [00:00<00:00, 3127.30it/s, Materializing param=encoder.layer.5.attention.self.value.bias]
Loading weights:  24%|██▍       | 95/391 [00:00<00:00, 3139.25it/s, Materializing param=encoder.layer.5.attention.self.value.weight]
Loading weights:  24%|██▍       | 95/391 [00:00<00:00, 3126.74it/s, Materializing param=encoder.layer.5.attention.self.value.weight]
Loading weights:  25%|██▍       | 96/391 [00:00<00:00, 3137.13it/s, Materializing param=encoder.layer.5.intermediate.dense.bias]    
Loading weights:  25%|██▍       | 96/391 [00:00<00:00, 3124.01it/s, Materializing param=encoder.layer.5.intermediate.dense.bias]
Loading weights:  25%|██▍       | 97/391 [00:00<00:00, 3133.62it/s, Materializing param=encoder.layer.5.intermediate.dense.weight]
Loading weights:  25%|██▍       | 97/391 [00:00<00:00, 3120.38it/s, Materializing param=encoder.layer.5.intermediate.dense.weight]
Loading weights:  25%|██▌       | 98/391 [00:00<00:00, 3132.87it/s, Materializing param=encoder.layer.5.output.LayerNorm.bias]    
Loading weights:  25%|██▌       | 98/391 [00:00<00:00, 3120.93it/s, Materializing param=encoder.layer.5.output.LayerNorm.bias]
Loading weights:  25%|██▌       | 99/391 [00:00<00:00, 3133.88it/s, Materializing param=encoder.layer.5.output.LayerNorm.weight]
Loading weights:  25%|██▌       | 99/391 [00:00<00:00, 3122.52it/s, Materializing param=encoder.layer.5.output.LayerNorm.weight]
Loading weights:  26%|██▌       | 100/391 [00:00<00:00, 3135.60it/s, Materializing param=encoder.layer.5.output.dense.bias]     
Loading weights:  26%|██▌       | 100/391 [00:00<00:00, 3124.64it/s, Materializing param=encoder.layer.5.output.dense.bias]
Loading weights:  26%|██▌       | 101/391 [00:00<00:00, 3138.36it/s, Materializing param=encoder.layer.5.output.dense.weight]
Loading weights:  26%|██▌       | 101/391 [00:00<00:00, 3128.04it/s, Materializing param=encoder.layer.5.output.dense.weight]
Loading weights:  26%|██▌       | 102/391 [00:00<00:00, 3142.86it/s, Materializing param=encoder.layer.6.attention.output.LayerNorm.bias]
Loading weights:  26%|██▌       | 102/391 [00:00<00:00, 3132.46it/s, Materializing param=encoder.layer.6.attention.output.LayerNorm.bias]
Loading weights:  26%|██▋       | 103/391 [00:00<00:00, 3146.61it/s, Materializing param=encoder.layer.6.attention.output.LayerNorm.weight]
Loading weights:  26%|██▋       | 103/391 [00:00<00:00, 3135.05it/s, Materializing param=encoder.layer.6.attention.output.LayerNorm.weight]
Loading weights:  27%|██▋       | 104/391 [00:00<00:00, 3148.01it/s, Materializing param=encoder.layer.6.attention.output.dense.bias]      
Loading weights:  27%|██▋       | 104/391 [00:00<00:00, 3136.97it/s, Materializing param=encoder.layer.6.attention.output.dense.bias]
Loading weights:  27%|██▋       | 105/391 [00:00<00:00, 3150.93it/s, Materializing param=encoder.layer.6.attention.output.dense.weight]
Loading weights:  27%|██▋       | 105/391 [00:00<00:00, 3140.88it/s, Materializing param=encoder.layer.6.attention.output.dense.weight]
Loading weights:  27%|██▋       | 106/391 [00:00<00:00, 3155.16it/s, Materializing param=encoder.layer.6.attention.self.key.bias]      
Loading weights:  27%|██▋       | 106/391 [00:00<00:00, 3145.33it/s, Materializing param=encoder.layer.6.attention.self.key.bias]
Loading weights:  27%|██▋       | 107/391 [00:00<00:00, 3157.52it/s, Materializing param=encoder.layer.6.attention.self.key.weight]
Loading weights:  27%|██▋       | 107/391 [00:00<00:00, 3145.63it/s, Materializing param=encoder.layer.6.attention.self.key.weight]
Loading weights:  28%|██▊       | 108/391 [00:00<00:00, 3156.18it/s, Materializing param=encoder.layer.6.attention.self.query.bias]
Loading weights:  28%|██▊       | 108/391 [00:00<00:00, 3145.25it/s, Materializing param=encoder.layer.6.attention.self.query.bias]
Loading weights:  28%|██▊       | 109/391 [00:00<00:00, 3156.99it/s, Materializing param=encoder.layer.6.attention.self.query.weight]
Loading weights:  28%|██▊       | 109/391 [00:00<00:00, 3143.98it/s, Materializing param=encoder.layer.6.attention.self.query.weight]
Loading weights:  28%|██▊       | 110/391 [00:00<00:00, 3153.03it/s, Materializing param=encoder.layer.6.attention.self.value.bias]  
Loading weights:  28%|██▊       | 110/391 [00:00<00:00, 3140.39it/s, Materializing param=encoder.layer.6.attention.self.value.bias]
Loading weights:  28%|██▊       | 111/391 [00:00<00:00, 3150.07it/s, Materializing param=encoder.layer.6.attention.self.value.weight]
Loading weights:  28%|██▊       | 111/391 [00:00<00:00, 3139.07it/s, Materializing param=encoder.layer.6.attention.self.value.weight]
Loading weights:  29%|██▊       | 112/391 [00:00<00:00, 3150.16it/s, Materializing param=encoder.layer.6.intermediate.dense.bias]    
Loading weights:  29%|██▊       | 112/391 [00:00<00:00, 3139.51it/s, Materializing param=encoder.layer.6.intermediate.dense.bias]
Loading weights:  29%|██▉       | 113/391 [00:00<00:00, 3149.02it/s, Materializing param=encoder.layer.6.intermediate.dense.weight]
Loading weights:  29%|██▉       | 113/391 [00:00<00:00, 3137.70it/s, Materializing param=encoder.layer.6.intermediate.dense.weight]
Loading weights:  29%|██▉       | 114/391 [00:00<00:00, 3147.94it/s, Materializing param=encoder.layer.6.output.LayerNorm.bias]    
Loading weights:  29%|██▉       | 114/391 [00:00<00:00, 3137.55it/s, Materializing param=encoder.layer.6.output.LayerNorm.bias]
Loading weights:  29%|██▉       | 115/391 [00:00<00:00, 3148.61it/s, Materializing param=encoder.layer.6.output.LayerNorm.weight]
Loading weights:  29%|██▉       | 115/391 [00:00<00:00, 3138.33it/s, Materializing param=encoder.layer.6.output.LayerNorm.weight]
Loading weights:  30%|██▉       | 116/391 [00:00<00:00, 3149.28it/s, Materializing param=encoder.layer.6.output.dense.bias]      
Loading weights:  30%|██▉       | 116/391 [00:00<00:00, 3137.73it/s, Materializing param=encoder.layer.6.output.dense.bias]
Loading weights:  30%|██▉       | 117/391 [00:00<00:00, 3147.71it/s, Materializing param=encoder.layer.6.output.dense.weight]
Loading weights:  30%|██▉       | 117/391 [00:00<00:00, 3137.50it/s, Materializing param=encoder.layer.6.output.dense.weight]
Loading weights:  30%|███       | 118/391 [00:00<00:00, 3148.24it/s, Materializing param=encoder.layer.7.attention.output.LayerNorm.bias]
Loading weights:  30%|███       | 118/391 [00:00<00:00, 3137.94it/s, Materializing param=encoder.layer.7.attention.output.LayerNorm.bias]
Loading weights:  30%|███       | 119/391 [00:00<00:00, 3148.00it/s, Materializing param=encoder.layer.7.attention.output.LayerNorm.weight]
Loading weights:  30%|███       | 119/391 [00:00<00:00, 3136.59it/s, Materializing param=encoder.layer.7.attention.output.LayerNorm.weight]
Loading weights:  31%|███       | 120/391 [00:00<00:00, 3145.81it/s, Materializing param=encoder.layer.7.attention.output.dense.bias]      
Loading weights:  31%|███       | 120/391 [00:00<00:00, 3135.83it/s, Materializing param=encoder.layer.7.attention.output.dense.bias]
Loading weights:  31%|███       | 121/391 [00:00<00:00, 3146.05it/s, Materializing param=encoder.layer.7.attention.output.dense.weight]
Loading weights:  31%|███       | 121/391 [00:00<00:00, 3136.29it/s, Materializing param=encoder.layer.7.attention.output.dense.weight]
Loading weights:  31%|███       | 122/391 [00:00<00:00, 3146.65it/s, Materializing param=encoder.layer.7.attention.self.key.bias]      
Loading weights:  31%|███       | 122/391 [00:00<00:00, 3134.64it/s, Materializing param=encoder.layer.7.attention.self.key.bias]
Loading weights:  31%|███▏      | 123/391 [00:00<00:00, 3143.24it/s, Materializing param=encoder.layer.7.attention.self.key.weight]
Loading weights:  31%|███▏      | 123/391 [00:00<00:00, 3133.52it/s, Materializing param=encoder.layer.7.attention.self.key.weight]
Loading weights:  32%|███▏      | 124/391 [00:00<00:00, 3143.59it/s, Materializing param=encoder.layer.7.attention.self.query.bias]
Loading weights:  32%|███▏      | 124/391 [00:00<00:00, 3133.93it/s, Materializing param=encoder.layer.7.attention.self.query.bias]
Loading weights:  32%|███▏      | 125/391 [00:00<00:00, 3143.87it/s, Materializing param=encoder.layer.7.attention.self.query.weight]
Loading weights:  32%|███▏      | 125/391 [00:00<00:00, 3133.30it/s, Materializing param=encoder.layer.7.attention.self.query.weight]
Loading weights:  32%|███▏      | 126/391 [00:00<00:00, 3141.82it/s, Materializing param=encoder.layer.7.attention.self.value.bias]  
Loading weights:  32%|███▏      | 126/391 [00:00<00:00, 3132.12it/s, Materializing param=encoder.layer.7.attention.self.value.bias]
Loading weights:  32%|███▏      | 127/391 [00:00<00:00, 3141.78it/s, Materializing param=encoder.layer.7.attention.self.value.weight]
Loading weights:  32%|███▏      | 127/391 [00:00<00:00, 3132.80it/s, Materializing param=encoder.layer.7.attention.self.value.weight]
Loading weights:  33%|███▎      | 128/391 [00:00<00:00, 3144.01it/s, Materializing param=encoder.layer.7.intermediate.dense.bias]    
Loading weights:  33%|███▎      | 128/391 [00:00<00:00, 3135.80it/s, Materializing param=encoder.layer.7.intermediate.dense.bias]
Loading weights:  33%|███▎      | 129/391 [00:00<00:00, 3145.49it/s, Materializing param=encoder.layer.7.intermediate.dense.weight]
Loading weights:  33%|███▎      | 129/391 [00:00<00:00, 3136.61it/s, Materializing param=encoder.layer.7.intermediate.dense.weight]
Loading weights:  33%|███▎      | 130/391 [00:00<00:00, 3148.13it/s, Materializing param=encoder.layer.7.output.LayerNorm.bias]    
Loading weights:  33%|███▎      | 130/391 [00:00<00:00, 3140.12it/s, Materializing param=encoder.layer.7.output.LayerNorm.bias]
Loading weights:  34%|███▎      | 131/391 [00:00<00:00, 3151.53it/s, Materializing param=encoder.layer.7.output.LayerNorm.weight]
Loading weights:  34%|███▎      | 131/391 [00:00<00:00, 3143.56it/s, Materializing param=encoder.layer.7.output.LayerNorm.weight]
Loading weights:  34%|███▍      | 132/391 [00:00<00:00, 3154.29it/s, Materializing param=encoder.layer.7.output.dense.bias]      
Loading weights:  34%|███▍      | 132/391 [00:00<00:00, 3145.98it/s, Materializing param=encoder.layer.7.output.dense.bias]
Loading weights:  34%|███▍      | 133/391 [00:00<00:00, 3153.75it/s, Materializing param=encoder.layer.7.output.dense.weight]
Loading weights:  34%|███▍      | 133/391 [00:00<00:00, 3145.89it/s, Materializing param=encoder.layer.7.output.dense.weight]
Loading weights:  34%|███▍      | 134/391 [00:00<00:00, 3157.30it/s, Materializing param=encoder.layer.8.attention.output.LayerNorm.bias]
Loading weights:  34%|███▍      | 134/391 [00:00<00:00, 3149.28it/s, Materializing param=encoder.layer.8.attention.output.LayerNorm.bias]
Loading weights:  35%|███▍      | 135/391 [00:00<00:00, 3160.04it/s, Materializing param=encoder.layer.8.attention.output.LayerNorm.weight]
Loading weights:  35%|███▍      | 135/391 [00:00<00:00, 3151.93it/s, Materializing param=encoder.layer.8.attention.output.LayerNorm.weight]
Loading weights:  35%|███▍      | 136/391 [00:00<00:00, 3159.08it/s, Materializing param=encoder.layer.8.attention.output.dense.bias]      
Loading weights:  35%|███▍      | 136/391 [00:00<00:00, 3149.43it/s, Materializing param=encoder.layer.8.attention.output.dense.bias]
Loading weights:  35%|███▌      | 137/391 [00:00<00:00, 3158.36it/s, Materializing param=encoder.layer.8.attention.output.dense.weight]
Loading weights:  35%|███▌      | 137/391 [00:00<00:00, 3149.57it/s, Materializing param=encoder.layer.8.attention.output.dense.weight]
Loading weights:  35%|███▌      | 138/391 [00:00<00:00, 3158.50it/s, Materializing param=encoder.layer.8.attention.self.key.bias]      
Loading weights:  35%|███▌      | 138/391 [00:00<00:00, 3149.82it/s, Materializing param=encoder.layer.8.attention.self.key.bias]
Loading weights:  36%|███▌      | 139/391 [00:00<00:00, 3156.92it/s, Materializing param=encoder.layer.8.attention.self.key.weight]
Loading weights:  36%|███▌      | 139/391 [00:00<00:00, 3147.57it/s, Materializing param=encoder.layer.8.attention.self.key.weight]
Loading weights:  36%|███▌      | 140/391 [00:00<00:00, 3156.10it/s, Materializing param=encoder.layer.8.attention.self.query.bias]
Loading weights:  36%|███▌      | 140/391 [00:00<00:00, 3147.48it/s, Materializing param=encoder.layer.8.attention.self.query.bias]
Loading weights:  36%|███▌      | 141/391 [00:00<00:00, 3156.15it/s, Materializing param=encoder.layer.8.attention.self.query.weight]
Loading weights:  36%|███▌      | 141/391 [00:00<00:00, 3147.54it/s, Materializing param=encoder.layer.8.attention.self.query.weight]
Loading weights:  36%|███▋      | 142/391 [00:00<00:00, 3154.58it/s, Materializing param=encoder.layer.8.attention.self.value.bias]  
Loading weights:  36%|███▋      | 142/391 [00:00<00:00, 3145.42it/s, Materializing param=encoder.layer.8.attention.self.value.bias]
Loading weights:  37%|███▋      | 143/391 [00:00<00:00, 3153.81it/s, Materializing param=encoder.layer.8.attention.self.value.weight]
Loading weights:  37%|███▋      | 143/391 [00:00<00:00, 3145.39it/s, Materializing param=encoder.layer.8.attention.self.value.weight]
Loading weights:  37%|███▋      | 144/391 [00:00<00:00, 3154.09it/s, Materializing param=encoder.layer.8.intermediate.dense.bias]    
Loading weights:  37%|███▋      | 144/391 [00:00<00:00, 3145.73it/s, Materializing param=encoder.layer.8.intermediate.dense.bias]
Loading weights:  37%|███▋      | 145/391 [00:00<00:00, 3153.06it/s, Materializing param=encoder.layer.8.intermediate.dense.weight]
Loading weights:  37%|███▋      | 145/391 [00:00<00:00, 3144.04it/s, Materializing param=encoder.layer.8.intermediate.dense.weight]
Loading weights:  37%|███▋      | 146/391 [00:00<00:00, 3152.39it/s, Materializing param=encoder.layer.8.output.LayerNorm.bias]    
Loading weights:  37%|███▋      | 146/391 [00:00<00:00, 3144.11it/s, Materializing param=encoder.layer.8.output.LayerNorm.bias]
Loading weights:  38%|███▊      | 147/391 [00:00<00:00, 3152.50it/s, Materializing param=encoder.layer.8.output.LayerNorm.weight]
Loading weights:  38%|███▊      | 147/391 [00:00<00:00, 3143.76it/s, Materializing param=encoder.layer.8.output.LayerNorm.weight]
Loading weights:  38%|███▊      | 148/391 [00:00<00:00, 3151.15it/s, Materializing param=encoder.layer.8.output.dense.bias]      
Loading weights:  38%|███▊      | 148/391 [00:00<00:00, 3142.55it/s, Materializing param=encoder.layer.8.output.dense.bias]
Loading weights:  38%|███▊      | 149/391 [00:00<00:00, 3150.80it/s, Materializing param=encoder.layer.8.output.dense.weight]
Loading weights:  38%|███▊      | 149/391 [00:00<00:00, 3142.89it/s, Materializing param=encoder.layer.8.output.dense.weight]
Loading weights:  38%|███▊      | 150/391 [00:00<00:00, 3151.50it/s, Materializing param=encoder.layer.9.attention.output.LayerNorm.bias]
Loading weights:  38%|███▊      | 150/391 [00:00<00:00, 3143.37it/s, Materializing param=encoder.layer.9.attention.output.LayerNorm.bias]
Loading weights:  39%|███▊      | 151/391 [00:00<00:00, 3150.21it/s, Materializing param=encoder.layer.9.attention.output.LayerNorm.weight]
Loading weights:  39%|███▊      | 151/391 [00:00<00:00, 3141.85it/s, Materializing param=encoder.layer.9.attention.output.LayerNorm.weight]
Loading weights:  39%|███▉      | 152/391 [00:00<00:00, 3149.20it/s, Materializing param=encoder.layer.9.attention.output.dense.bias]      
Loading weights:  39%|███▉      | 152/391 [00:00<00:00, 3141.26it/s, Materializing param=encoder.layer.9.attention.output.dense.bias]
Loading weights:  39%|███▉      | 153/391 [00:00<00:00, 3149.33it/s, Materializing param=encoder.layer.9.attention.output.dense.weight]
Loading weights:  39%|███▉      | 153/391 [00:00<00:00, 3141.59it/s, Materializing param=encoder.layer.9.attention.output.dense.weight]
Loading weights:  39%|███▉      | 154/391 [00:00<00:00, 3148.62it/s, Materializing param=encoder.layer.9.attention.self.key.bias]      
Loading weights:  39%|███▉      | 154/391 [00:00<00:00, 3140.62it/s, Materializing param=encoder.layer.9.attention.self.key.bias]
Loading weights:  40%|███▉      | 155/391 [00:00<00:00, 3147.75it/s, Materializing param=encoder.layer.9.attention.self.key.weight]
Loading weights:  40%|███▉      | 155/391 [00:00<00:00, 3140.01it/s, Materializing param=encoder.layer.9.attention.self.key.weight]
Loading weights:  40%|███▉      | 156/391 [00:00<00:00, 3148.04it/s, Materializing param=encoder.layer.9.attention.self.query.bias]
Loading weights:  40%|███▉      | 156/391 [00:00<00:00, 3140.52it/s, Materializing param=encoder.layer.9.attention.self.query.bias]
Loading weights:  40%|████      | 157/391 [00:00<00:00, 3147.39it/s, Materializing param=encoder.layer.9.attention.self.query.weight]
Loading weights:  40%|████      | 157/391 [00:00<00:00, 3139.54it/s, Materializing param=encoder.layer.9.attention.self.query.weight]
Loading weights:  40%|████      | 158/391 [00:00<00:00, 3146.66it/s, Materializing param=encoder.layer.9.attention.self.value.bias]  
Loading weights:  40%|████      | 158/391 [00:00<00:00, 3139.42it/s, Materializing param=encoder.layer.9.attention.self.value.bias]
Loading weights:  41%|████      | 159/391 [00:00<00:00, 3148.22it/s, Materializing param=encoder.layer.9.attention.self.value.weight]
Loading weights:  41%|████      | 159/391 [00:00<00:00, 3141.48it/s, Materializing param=encoder.layer.9.attention.self.value.weight]
Loading weights:  41%|████      | 160/391 [00:00<00:00, 3150.71it/s, Materializing param=encoder.layer.9.intermediate.dense.bias]    
Loading weights:  41%|████      | 160/391 [00:00<00:00, 3143.21it/s, Materializing param=encoder.layer.9.intermediate.dense.bias]
Loading weights:  41%|████      | 161/391 [00:00<00:00, 3150.02it/s, Materializing param=encoder.layer.9.intermediate.dense.weight]
Loading weights:  41%|████      | 161/391 [00:00<00:00, 3142.50it/s, Materializing param=encoder.layer.9.intermediate.dense.weight]
Loading weights:  41%|████▏     | 162/391 [00:00<00:00, 3150.21it/s, Materializing param=encoder.layer.9.output.LayerNorm.bias]    
Loading weights:  41%|████▏     | 162/391 [00:00<00:00, 3142.93it/s, Materializing param=encoder.layer.9.output.LayerNorm.bias]
Loading weights:  42%|████▏     | 163/391 [00:00<00:00, 3150.76it/s, Materializing param=encoder.layer.9.output.LayerNorm.weight]
Loading weights:  42%|████▏     | 163/391 [00:00<00:00, 3142.73it/s, Materializing param=encoder.layer.9.output.LayerNorm.weight]
Loading weights:  42%|████▏     | 164/391 [00:00<00:00, 3149.53it/s, Materializing param=encoder.layer.9.output.dense.bias]      
Loading weights:  42%|████▏     | 164/391 [00:00<00:00, 3142.29it/s, Materializing param=encoder.layer.9.output.dense.bias]
Loading weights:  42%|████▏     | 165/391 [00:00<00:00, 3149.99it/s, Materializing param=encoder.layer.9.output.dense.weight]
Loading weights:  42%|████▏     | 165/391 [00:00<00:00, 3142.96it/s, Materializing param=encoder.layer.9.output.dense.weight]
Loading weights:  42%|████▏     | 166/391 [00:00<00:00, 3151.17it/s, Materializing param=encoder.layer.10.attention.output.LayerNorm.bias]
Loading weights:  42%|████▏     | 166/391 [00:00<00:00, 3144.61it/s, Materializing param=encoder.layer.10.attention.output.LayerNorm.bias]
Loading weights:  43%|████▎     | 167/391 [00:00<00:00, 3151.67it/s, Materializing param=encoder.layer.10.attention.output.LayerNorm.weight]
Loading weights:  43%|████▎     | 167/391 [00:00<00:00, 3144.37it/s, Materializing param=encoder.layer.10.attention.output.LayerNorm.weight]
Loading weights:  43%|████▎     | 168/391 [00:00<00:00, 3152.65it/s, Materializing param=encoder.layer.10.attention.output.dense.bias]      
Loading weights:  43%|████▎     | 168/391 [00:00<00:00, 3146.28it/s, Materializing param=encoder.layer.10.attention.output.dense.bias]
Loading weights:  43%|████▎     | 169/391 [00:00<00:00, 3154.85it/s, Materializing param=encoder.layer.10.attention.output.dense.weight]
Loading weights:  43%|████▎     | 169/391 [00:00<00:00, 3148.51it/s, Materializing param=encoder.layer.10.attention.output.dense.weight]
Loading weights:  43%|████▎     | 170/391 [00:00<00:00, 3156.31it/s, Materializing param=encoder.layer.10.attention.self.key.bias]      
Loading weights:  43%|████▎     | 170/391 [00:00<00:00, 3148.40it/s, Materializing param=encoder.layer.10.attention.self.key.bias]
Loading weights:  44%|████▎     | 171/391 [00:00<00:00, 3155.37it/s, Materializing param=encoder.layer.10.attention.self.key.weight]
Loading weights:  44%|████▎     | 171/391 [00:00<00:00, 3148.70it/s, Materializing param=encoder.layer.10.attention.self.key.weight]
Loading weights:  44%|████▍     | 172/391 [00:00<00:00, 3156.74it/s, Materializing param=encoder.layer.10.attention.self.query.bias]
Loading weights:  44%|████▍     | 172/391 [00:00<00:00, 3150.18it/s, Materializing param=encoder.layer.10.attention.self.query.bias]
Loading weights:  44%|████▍     | 173/391 [00:00<00:00, 3158.09it/s, Materializing param=encoder.layer.10.attention.self.query.weight]
Loading weights:  44%|████▍     | 173/391 [00:00<00:00, 3150.76it/s, Materializing param=encoder.layer.10.attention.self.query.weight]
Loading weights:  45%|████▍     | 174/391 [00:00<00:00, 3155.90it/s, Materializing param=encoder.layer.10.attention.self.value.bias]  
Loading weights:  45%|████▍     | 174/391 [00:00<00:00, 3148.12it/s, Materializing param=encoder.layer.10.attention.self.value.bias]
Loading weights:  45%|████▍     | 175/391 [00:00<00:00, 3155.05it/s, Materializing param=encoder.layer.10.attention.self.value.weight]
Loading weights:  45%|████▍     | 175/391 [00:00<00:00, 3148.09it/s, Materializing param=encoder.layer.10.attention.self.value.weight]
Loading weights:  45%|████▌     | 176/391 [00:00<00:00, 3155.14it/s, Materializing param=encoder.layer.10.intermediate.dense.bias]    
Loading weights:  45%|████▌     | 176/391 [00:00<00:00, 3147.33it/s, Materializing param=encoder.layer.10.intermediate.dense.bias]
Loading weights:  45%|████▌     | 177/391 [00:00<00:00, 3153.57it/s, Materializing param=encoder.layer.10.intermediate.dense.weight]
Loading weights:  45%|████▌     | 177/391 [00:00<00:00, 3146.67it/s, Materializing param=encoder.layer.10.intermediate.dense.weight]
Loading weights:  46%|████▌     | 178/391 [00:00<00:00, 3153.80it/s, Materializing param=encoder.layer.10.output.LayerNorm.bias]    
Loading weights:  46%|████▌     | 178/391 [00:00<00:00, 3147.22it/s, Materializing param=encoder.layer.10.output.LayerNorm.bias]
Loading weights:  46%|████▌     | 179/391 [00:00<00:00, 3154.21it/s, Materializing param=encoder.layer.10.output.LayerNorm.weight]
Loading weights:  46%|████▌     | 179/391 [00:00<00:00, 3147.00it/s, Materializing param=encoder.layer.10.output.LayerNorm.weight]
Loading weights:  46%|████▌     | 180/391 [00:00<00:00, 3152.98it/s, Materializing param=encoder.layer.10.output.dense.bias]      
Loading weights:  46%|████▌     | 180/391 [00:00<00:00, 3146.29it/s, Materializing param=encoder.layer.10.output.dense.bias]
Loading weights:  46%|████▋     | 181/391 [00:00<00:00, 3153.17it/s, Materializing param=encoder.layer.10.output.dense.weight]
Loading weights:  46%|████▋     | 181/391 [00:00<00:00, 3146.61it/s, Materializing param=encoder.layer.10.output.dense.weight]
Loading weights:  47%|████▋     | 182/391 [00:00<00:00, 3153.56it/s, Materializing param=encoder.layer.11.attention.output.LayerNorm.bias]
Loading weights:  47%|████▋     | 182/391 [00:00<00:00, 3146.14it/s, Materializing param=encoder.layer.11.attention.output.LayerNorm.bias]
Loading weights:  47%|████▋     | 183/391 [00:00<00:00, 3152.07it/s, Materializing param=encoder.layer.11.attention.output.LayerNorm.weight]
Loading weights:  47%|████▋     | 183/391 [00:00<00:00, 3144.83it/s, Materializing param=encoder.layer.11.attention.output.LayerNorm.weight]
Loading weights:  47%|████▋     | 184/391 [00:00<00:00, 3151.13it/s, Materializing param=encoder.layer.11.attention.output.dense.bias]      
Loading weights:  47%|████▋     | 184/391 [00:00<00:00, 3144.54it/s, Materializing param=encoder.layer.11.attention.output.dense.bias]
Loading weights:  47%|████▋     | 185/391 [00:00<00:00, 3150.97it/s, Materializing param=encoder.layer.11.attention.output.dense.weight]
Loading weights:  47%|████▋     | 185/391 [00:00<00:00, 3143.19it/s, Materializing param=encoder.layer.11.attention.output.dense.weight]
Loading weights:  48%|████▊     | 186/391 [00:00<00:00, 3148.51it/s, Materializing param=encoder.layer.11.attention.self.key.bias]      
Loading weights:  48%|████▊     | 186/391 [00:00<00:00, 3141.85it/s, Materializing param=encoder.layer.11.attention.self.key.bias]
Loading weights:  48%|████▊     | 187/391 [00:00<00:00, 3148.28it/s, Materializing param=encoder.layer.11.attention.self.key.weight]
Loading weights:  48%|████▊     | 187/391 [00:00<00:00, 3141.90it/s, Materializing param=encoder.layer.11.attention.self.key.weight]
Loading weights:  48%|████▊     | 188/391 [00:00<00:00, 3148.52it/s, Materializing param=encoder.layer.11.attention.self.query.bias]
Loading weights:  48%|████▊     | 188/391 [00:00<00:00, 3142.14it/s, Materializing param=encoder.layer.11.attention.self.query.bias]
Loading weights:  48%|████▊     | 189/391 [00:00<00:00, 3147.39it/s, Materializing param=encoder.layer.11.attention.self.query.weight]
Loading weights:  48%|████▊     | 189/391 [00:00<00:00, 3140.54it/s, Materializing param=encoder.layer.11.attention.self.query.weight]
Loading weights:  49%|████▊     | 190/391 [00:00<00:00, 3146.49it/s, Materializing param=encoder.layer.11.attention.self.value.bias]  
Loading weights:  49%|████▊     | 190/391 [00:00<00:00, 3140.20it/s, Materializing param=encoder.layer.11.attention.self.value.bias]
Loading weights:  49%|████▉     | 191/391 [00:00<00:00, 3146.79it/s, Materializing param=encoder.layer.11.attention.self.value.weight]
Loading weights:  49%|████▉     | 191/391 [00:00<00:00, 3140.54it/s, Materializing param=encoder.layer.11.attention.self.value.weight]
Loading weights:  49%|████▉     | 192/391 [00:00<00:00, 3145.86it/s, Materializing param=encoder.layer.11.intermediate.dense.bias]    
Loading weights:  49%|████▉     | 192/391 [00:00<00:00, 3139.20it/s, Materializing param=encoder.layer.11.intermediate.dense.bias]
Loading weights:  49%|████▉     | 193/391 [00:00<00:00, 3145.60it/s, Materializing param=encoder.layer.11.intermediate.dense.weight]
Loading weights:  49%|████▉     | 193/391 [00:00<00:00, 3139.38it/s, Materializing param=encoder.layer.11.intermediate.dense.weight]
Loading weights:  50%|████▉     | 194/391 [00:00<00:00, 3146.00it/s, Materializing param=encoder.layer.11.output.LayerNorm.bias]    
Loading weights:  50%|████▉     | 194/391 [00:00<00:00, 3139.87it/s, Materializing param=encoder.layer.11.output.LayerNorm.bias]
Loading weights:  50%|████▉     | 195/391 [00:00<00:00, 3145.39it/s, Materializing param=encoder.layer.11.output.LayerNorm.weight]
Loading weights:  50%|████▉     | 195/391 [00:00<00:00, 3137.15it/s, Materializing param=encoder.layer.11.output.LayerNorm.weight]
Loading weights:  50%|█████     | 196/391 [00:00<00:00, 3143.57it/s, Materializing param=encoder.layer.11.output.dense.bias]      
Loading weights:  50%|█████     | 196/391 [00:00<00:00, 3138.06it/s, Materializing param=encoder.layer.11.output.dense.bias]
Loading weights:  50%|█████     | 197/391 [00:00<00:00, 3145.59it/s, Materializing param=encoder.layer.11.output.dense.weight]
Loading weights:  50%|█████     | 197/391 [00:00<00:00, 3140.33it/s, Materializing param=encoder.layer.11.output.dense.weight]
Loading weights:  51%|█████     | 198/391 [00:00<00:00, 3146.98it/s, Materializing param=encoder.layer.12.attention.output.LayerNorm.bias]
Loading weights:  51%|█████     | 198/391 [00:00<00:00, 3141.24it/s, Materializing param=encoder.layer.12.attention.output.LayerNorm.bias]
Loading weights:  51%|█████     | 199/391 [00:00<00:00, 3147.71it/s, Materializing param=encoder.layer.12.attention.output.LayerNorm.weight]
Loading weights:  51%|█████     | 199/391 [00:00<00:00, 3142.19it/s, Materializing param=encoder.layer.12.attention.output.LayerNorm.weight]
Loading weights:  51%|█████     | 200/391 [00:00<00:00, 3149.47it/s, Materializing param=encoder.layer.12.attention.output.dense.bias]      
Loading weights:  51%|█████     | 200/391 [00:00<00:00, 3144.07it/s, Materializing param=encoder.layer.12.attention.output.dense.bias]
Loading weights:  51%|█████▏    | 201/391 [00:00<00:00, 3151.43it/s, Materializing param=encoder.layer.12.attention.output.dense.weight]
Loading weights:  51%|█████▏    | 201/391 [00:00<00:00, 3145.42it/s, Materializing param=encoder.layer.12.attention.output.dense.weight]
Loading weights:  52%|█████▏    | 202/391 [00:00<00:00, 3150.75it/s, Materializing param=encoder.layer.12.attention.self.key.bias]      
Loading weights:  52%|█████▏    | 202/391 [00:00<00:00, 3144.79it/s, Materializing param=encoder.layer.12.attention.self.key.bias]
Loading weights:  52%|█████▏    | 203/391 [00:00<00:00, 3150.89it/s, Materializing param=encoder.layer.12.attention.self.key.weight]
Loading weights:  52%|█████▏    | 203/391 [00:00<00:00, 3145.03it/s, Materializing param=encoder.layer.12.attention.self.key.weight]
Loading weights:  52%|█████▏    | 204/391 [00:00<00:00, 3151.03it/s, Materializing param=encoder.layer.12.attention.self.query.bias]
Loading weights:  52%|█████▏    | 204/391 [00:00<00:00, 3144.54it/s, Materializing param=encoder.layer.12.attention.self.query.bias]
Loading weights:  52%|█████▏    | 205/391 [00:00<00:00, 3149.73it/s, Materializing param=encoder.layer.12.attention.self.query.weight]
Loading weights:  52%|█████▏    | 205/391 [00:00<00:00, 3143.80it/s, Materializing param=encoder.layer.12.attention.self.query.weight]
Loading weights:  53%|█████▎    | 206/391 [00:00<00:00, 3149.70it/s, Materializing param=encoder.layer.12.attention.self.value.bias]  
Loading weights:  53%|█████▎    | 206/391 [00:00<00:00, 3143.95it/s, Materializing param=encoder.layer.12.attention.self.value.bias]
Loading weights:  53%|█████▎    | 207/391 [00:00<00:00, 3150.08it/s, Materializing param=encoder.layer.12.attention.self.value.weight]
Loading weights:  53%|█████▎    | 207/391 [00:00<00:00, 3144.30it/s, Materializing param=encoder.layer.12.attention.self.value.weight]
Loading weights:  53%|█████▎    | 208/391 [00:00<00:00, 3149.25it/s, Materializing param=encoder.layer.12.intermediate.dense.bias]    
Loading weights:  53%|█████▎    | 208/391 [00:00<00:00, 3143.15it/s, Materializing param=encoder.layer.12.intermediate.dense.bias]
Loading weights:  53%|█████▎    | 209/391 [00:00<00:00, 3149.15it/s, Materializing param=encoder.layer.12.intermediate.dense.weight]
Loading weights:  53%|█████▎    | 209/391 [00:00<00:00, 3143.50it/s, Materializing param=encoder.layer.12.intermediate.dense.weight]
Loading weights:  54%|█████▎    | 210/391 [00:00<00:00, 3149.60it/s, Materializing param=encoder.layer.12.output.LayerNorm.bias]    
Loading weights:  54%|█████▎    | 210/391 [00:00<00:00, 3143.99it/s, Materializing param=encoder.layer.12.output.LayerNorm.bias]
Loading weights:  54%|█████▍    | 211/391 [00:00<00:00, 3149.04it/s, Materializing param=encoder.layer.12.output.LayerNorm.weight]
Loading weights:  54%|█████▍    | 211/391 [00:00<00:00, 3143.01it/s, Materializing param=encoder.layer.12.output.LayerNorm.weight]
Loading weights:  54%|█████▍    | 212/391 [00:00<00:00, 3148.89it/s, Materializing param=encoder.layer.12.output.dense.bias]      
Loading weights:  54%|█████▍    | 212/391 [00:00<00:00, 3143.38it/s, Materializing param=encoder.layer.12.output.dense.bias]
Loading weights:  54%|█████▍    | 213/391 [00:00<00:00, 3149.31it/s, Materializing param=encoder.layer.12.output.dense.weight]
Loading weights:  54%|█████▍    | 213/391 [00:00<00:00, 3143.72it/s, Materializing param=encoder.layer.12.output.dense.weight]
Loading weights:  55%|█████▍    | 214/391 [00:00<00:00, 3148.98it/s, Materializing param=encoder.layer.13.attention.output.LayerNorm.bias]
Loading weights:  55%|█████▍    | 214/391 [00:00<00:00, 3142.86it/s, Materializing param=encoder.layer.13.attention.output.LayerNorm.bias]
Loading weights:  55%|█████▍    | 215/391 [00:00<00:00, 3148.27it/s, Materializing param=encoder.layer.13.attention.output.LayerNorm.weight]
Loading weights:  55%|█████▍    | 215/391 [00:00<00:00, 3142.58it/s, Materializing param=encoder.layer.13.attention.output.LayerNorm.weight]
Loading weights:  55%|█████▌    | 216/391 [00:00<00:00, 3148.23it/s, Materializing param=encoder.layer.13.attention.output.dense.bias]      
Loading weights:  55%|█████▌    | 216/391 [00:00<00:00, 3142.67it/s, Materializing param=encoder.layer.13.attention.output.dense.bias]
Loading weights:  55%|█████▌    | 217/391 [00:00<00:00, 3147.33it/s, Materializing param=encoder.layer.13.attention.output.dense.weight]
Loading weights:  55%|█████▌    | 217/391 [00:00<00:00, 3141.29it/s, Materializing param=encoder.layer.13.attention.output.dense.weight]
Loading weights:  56%|█████▌    | 218/391 [00:00<00:00, 3146.94it/s, Materializing param=encoder.layer.13.attention.self.key.bias]      
Loading weights:  56%|█████▌    | 218/391 [00:00<00:00, 3141.85it/s, Materializing param=encoder.layer.13.attention.self.key.bias]
Loading weights:  56%|█████▌    | 219/391 [00:00<00:00, 3148.36it/s, Materializing param=encoder.layer.13.attention.self.key.weight]
Loading weights:  56%|█████▌    | 219/391 [00:00<00:00, 3143.46it/s, Materializing param=encoder.layer.13.attention.self.key.weight]
Loading weights:  56%|█████▋    | 220/391 [00:00<00:00, 3149.58it/s, Materializing param=encoder.layer.13.attention.self.query.bias]
Loading weights:  56%|█████▋    | 220/391 [00:00<00:00, 3144.46it/s, Materializing param=encoder.layer.13.attention.self.query.bias]
Loading weights:  57%|█████▋    | 221/391 [00:00<00:00, 3150.38it/s, Materializing param=encoder.layer.13.attention.self.query.weight]
Loading weights:  57%|█████▋    | 221/391 [00:00<00:00, 3145.46it/s, Materializing param=encoder.layer.13.attention.self.query.weight]
Loading weights:  57%|█████▋    | 222/391 [00:00<00:00, 3151.96it/s, Materializing param=encoder.layer.13.attention.self.value.bias]  
Loading weights:  57%|█████▋    | 222/391 [00:00<00:00, 3147.17it/s, Materializing param=encoder.layer.13.attention.self.value.bias]
Loading weights:  57%|█████▋    | 223/391 [00:00<00:00, 3153.73it/s, Materializing param=encoder.layer.13.attention.self.value.weight]
Loading weights:  57%|█████▋    | 223/391 [00:00<00:00, 3148.95it/s, Materializing param=encoder.layer.13.attention.self.value.weight]
Loading weights:  57%|█████▋    | 224/391 [00:00<00:00, 3153.50it/s, Materializing param=encoder.layer.13.intermediate.dense.bias]    
Loading weights:  57%|█████▋    | 224/391 [00:00<00:00, 3147.66it/s, Materializing param=encoder.layer.13.intermediate.dense.bias]
Loading weights:  58%|█████▊    | 225/391 [00:00<00:00, 3153.10it/s, Materializing param=encoder.layer.13.intermediate.dense.weight]
Loading weights:  58%|█████▊    | 225/391 [00:00<00:00, 3147.73it/s, Materializing param=encoder.layer.13.intermediate.dense.weight]
Loading weights:  58%|█████▊    | 226/391 [00:00<00:00, 3153.35it/s, Materializing param=encoder.layer.13.output.LayerNorm.bias]    
Loading weights:  58%|█████▊    | 226/391 [00:00<00:00, 3148.09it/s, Materializing param=encoder.layer.13.output.LayerNorm.bias]
Loading weights:  58%|█████▊    | 227/391 [00:00<00:00, 3152.88it/s, Materializing param=encoder.layer.13.output.LayerNorm.weight]
Loading weights:  58%|█████▊    | 227/391 [00:00<00:00, 3147.28it/s, Materializing param=encoder.layer.13.output.LayerNorm.weight]
Loading weights:  58%|█████▊    | 228/391 [00:00<00:00, 3152.83it/s, Materializing param=encoder.layer.13.output.dense.bias]      
Loading weights:  58%|█████▊    | 228/391 [00:00<00:00, 3147.64it/s, Materializing param=encoder.layer.13.output.dense.bias]
Loading weights:  59%|█████▊    | 229/391 [00:00<00:00, 3153.27it/s, Materializing param=encoder.layer.13.output.dense.weight]
Loading weights:  59%|█████▊    | 229/391 [00:00<00:00, 3148.12it/s, Materializing param=encoder.layer.13.output.dense.weight]
Loading weights:  59%|█████▉    | 230/391 [00:00<00:00, 3152.95it/s, Materializing param=encoder.layer.14.attention.output.LayerNorm.bias]
Loading weights:  59%|█████▉    | 230/391 [00:00<00:00, 3147.23it/s, Materializing param=encoder.layer.14.attention.output.LayerNorm.bias]
Loading weights:  59%|█████▉    | 231/391 [00:00<00:00, 3152.20it/s, Materializing param=encoder.layer.14.attention.output.LayerNorm.weight]
Loading weights:  59%|█████▉    | 231/391 [00:00<00:00, 3147.03it/s, Materializing param=encoder.layer.14.attention.output.LayerNorm.weight]
Loading weights:  59%|█████▉    | 232/391 [00:00<00:00, 3152.28it/s, Materializing param=encoder.layer.14.attention.output.dense.bias]      
Loading weights:  59%|█████▉    | 232/391 [00:00<00:00, 3147.08it/s, Materializing param=encoder.layer.14.attention.output.dense.bias]
Loading weights:  60%|█████▉    | 233/391 [00:00<00:00, 3151.57it/s, Materializing param=encoder.layer.14.attention.output.dense.weight]
Loading weights:  60%|█████▉    | 233/391 [00:00<00:00, 3146.20it/s, Materializing param=encoder.layer.14.attention.output.dense.weight]
Loading weights:  60%|█████▉    | 234/391 [00:00<00:00, 3151.06it/s, Materializing param=encoder.layer.14.attention.self.key.bias]      
Loading weights:  60%|█████▉    | 234/391 [00:00<00:00, 3145.94it/s, Materializing param=encoder.layer.14.attention.self.key.bias]
Loading weights:  60%|██████    | 235/391 [00:00<00:00, 3151.29it/s, Materializing param=encoder.layer.14.attention.self.key.weight]
Loading weights:  60%|██████    | 235/391 [00:00<00:00, 3146.12it/s, Materializing param=encoder.layer.14.attention.self.key.weight]
Loading weights:  60%|██████    | 236/391 [00:00<00:00, 3150.61it/s, Materializing param=encoder.layer.14.attention.self.query.bias]
Loading weights:  60%|██████    | 236/391 [00:00<00:00, 3145.33it/s, Materializing param=encoder.layer.14.attention.self.query.bias]
Loading weights:  61%|██████    | 237/391 [00:00<00:00, 3149.99it/s, Materializing param=encoder.layer.14.attention.self.query.weight]
Loading weights:  61%|██████    | 237/391 [00:00<00:00, 3144.86it/s, Materializing param=encoder.layer.14.attention.self.query.weight]
Loading weights:  61%|██████    | 238/391 [00:00<00:00, 3148.84it/s, Materializing param=encoder.layer.14.attention.self.value.bias]  
Loading weights:  61%|██████    | 238/391 [00:00<00:00, 3143.18it/s, Materializing param=encoder.layer.14.attention.self.value.bias]
Loading weights:  61%|██████    | 239/391 [00:00<00:00, 3147.61it/s, Materializing param=encoder.layer.14.attention.self.value.weight]
Loading weights:  61%|██████    | 239/391 [00:00<00:00, 3142.38it/s, Materializing param=encoder.layer.14.attention.self.value.weight]
Loading weights:  61%|██████▏   | 240/391 [00:00<00:00, 3147.12it/s, Materializing param=encoder.layer.14.intermediate.dense.bias]    
Loading weights:  61%|██████▏   | 240/391 [00:00<00:00, 3142.09it/s, Materializing param=encoder.layer.14.intermediate.dense.bias]
Loading weights:  62%|██████▏   | 241/391 [00:00<00:00, 3147.32it/s, Materializing param=encoder.layer.14.intermediate.dense.weight]
Loading weights:  62%|██████▏   | 241/391 [00:00<00:00, 3142.41it/s, Materializing param=encoder.layer.14.intermediate.dense.weight]
Loading weights:  62%|██████▏   | 242/391 [00:00<00:00, 3147.22it/s, Materializing param=encoder.layer.14.output.LayerNorm.bias]    
Loading weights:  62%|██████▏   | 242/391 [00:00<00:00, 3142.51it/s, Materializing param=encoder.layer.14.output.LayerNorm.bias]
Loading weights:  62%|██████▏   | 243/391 [00:00<00:00, 3148.05it/s, Materializing param=encoder.layer.14.output.LayerNorm.weight]
Loading weights:  62%|██████▏   | 243/391 [00:00<00:00, 3143.62it/s, Materializing param=encoder.layer.14.output.LayerNorm.weight]
Loading weights:  62%|██████▏   | 244/391 [00:00<00:00, 3149.73it/s, Materializing param=encoder.layer.14.output.dense.bias]      
Loading weights:  62%|██████▏   | 244/391 [00:00<00:00, 3145.49it/s, Materializing param=encoder.layer.14.output.dense.bias]
Loading weights:  63%|██████▎   | 245/391 [00:00<00:00, 3151.81it/s, Materializing param=encoder.layer.14.output.dense.weight]
Loading weights:  63%|██████▎   | 245/391 [00:00<00:00, 3147.09it/s, Materializing param=encoder.layer.14.output.dense.weight]
Loading weights:  63%|██████▎   | 246/391 [00:00<00:00, 3151.90it/s, Materializing param=encoder.layer.15.attention.output.LayerNorm.bias]
Loading weights:  63%|██████▎   | 246/391 [00:00<00:00, 3146.62it/s, Materializing param=encoder.layer.15.attention.output.LayerNorm.bias]
Loading weights:  63%|██████▎   | 247/391 [00:00<00:00, 3151.37it/s, Materializing param=encoder.layer.15.attention.output.LayerNorm.weight]
Loading weights:  63%|██████▎   | 247/391 [00:00<00:00, 3146.42it/s, Materializing param=encoder.layer.15.attention.output.LayerNorm.weight]
Loading weights:  63%|██████▎   | 248/391 [00:00<00:00, 3151.31it/s, Materializing param=encoder.layer.15.attention.output.dense.bias]      
Loading weights:  63%|██████▎   | 248/391 [00:00<00:00, 3146.05it/s, Materializing param=encoder.layer.15.attention.output.dense.bias]
Loading weights:  64%|██████▎   | 249/391 [00:00<00:00, 3149.75it/s, Materializing param=encoder.layer.15.attention.output.dense.weight]
Loading weights:  64%|██████▎   | 249/391 [00:00<00:00, 3144.46it/s, Materializing param=encoder.layer.15.attention.output.dense.weight]
Loading weights:  64%|██████▍   | 250/391 [00:00<00:00, 3149.22it/s, Materializing param=encoder.layer.15.attention.self.key.bias]      
Loading weights:  64%|██████▍   | 250/391 [00:00<00:00, 3144.41it/s, Materializing param=encoder.layer.15.attention.self.key.bias]
Loading weights:  64%|██████▍   | 251/391 [00:00<00:00, 3149.31it/s, Materializing param=encoder.layer.15.attention.self.key.weight]
Loading weights:  64%|██████▍   | 251/391 [00:00<00:00, 3144.51it/s, Materializing param=encoder.layer.15.attention.self.key.weight]
Loading weights:  64%|██████▍   | 252/391 [00:00<00:00, 3148.36it/s, Materializing param=encoder.layer.15.attention.self.query.bias]
Loading weights:  64%|██████▍   | 252/391 [00:00<00:00, 3143.17it/s, Materializing param=encoder.layer.15.attention.self.query.bias]
Loading weights:  65%|██████▍   | 253/391 [00:00<00:00, 3147.97it/s, Materializing param=encoder.layer.15.attention.self.query.weight]
Loading weights:  65%|██████▍   | 253/391 [00:00<00:00, 3143.22it/s, Materializing param=encoder.layer.15.attention.self.query.weight]
Loading weights:  65%|██████▍   | 254/391 [00:00<00:00, 3148.18it/s, Materializing param=encoder.layer.15.attention.self.value.bias]  
Loading weights:  65%|██████▍   | 254/391 [00:00<00:00, 3142.49it/s, Materializing param=encoder.layer.15.attention.self.value.bias]
Loading weights:  65%|██████▌   | 255/391 [00:00<00:00, 3144.78it/s, Materializing param=encoder.layer.15.attention.self.value.weight]
Loading weights:  65%|██████▌   | 255/391 [00:00<00:00, 3139.46it/s, Materializing param=encoder.layer.15.attention.self.value.weight]
Loading weights:  65%|██████▌   | 256/391 [00:00<00:00, 3143.27it/s, Materializing param=encoder.layer.15.intermediate.dense.bias]    
Loading weights:  65%|██████▌   | 256/391 [00:00<00:00, 3138.06it/s, Materializing param=encoder.layer.15.intermediate.dense.bias]
Loading weights:  66%|██████▌   | 257/391 [00:00<00:00, 3141.76it/s, Materializing param=encoder.layer.15.intermediate.dense.weight]
Loading weights:  66%|██████▌   | 257/391 [00:00<00:00, 3136.28it/s, Materializing param=encoder.layer.15.intermediate.dense.weight]
Loading weights:  66%|██████▌   | 258/391 [00:00<00:00, 3140.31it/s, Materializing param=encoder.layer.15.output.LayerNorm.bias]    
Loading weights:  66%|██████▌   | 258/391 [00:00<00:00, 3135.52it/s, Materializing param=encoder.layer.15.output.LayerNorm.bias]
Loading weights:  66%|██████▌   | 259/391 [00:00<00:00, 3141.21it/s, Materializing param=encoder.layer.15.output.LayerNorm.weight]
Loading weights:  66%|██████▌   | 259/391 [00:00<00:00, 3137.20it/s, Materializing param=encoder.layer.15.output.LayerNorm.weight]
Loading weights:  66%|██████▋   | 260/391 [00:00<00:00, 3143.11it/s, Materializing param=encoder.layer.15.output.dense.bias]      
Loading weights:  66%|██████▋   | 260/391 [00:00<00:00, 3139.20it/s, Materializing param=encoder.layer.15.output.dense.bias]
Loading weights:  67%|██████▋   | 261/391 [00:00<00:00, 3143.39it/s, Materializing param=encoder.layer.15.output.dense.weight]
Loading weights:  67%|██████▋   | 261/391 [00:00<00:00, 3138.57it/s, Materializing param=encoder.layer.15.output.dense.weight]
Loading weights:  67%|██████▋   | 262/391 [00:00<00:00, 3143.36it/s, Materializing param=encoder.layer.16.attention.output.LayerNorm.bias]
Loading weights:  67%|██████▋   | 262/391 [00:00<00:00, 3138.79it/s, Materializing param=encoder.layer.16.attention.output.LayerNorm.bias]
Loading weights:  67%|██████▋   | 263/391 [00:00<00:00, 3143.23it/s, Materializing param=encoder.layer.16.attention.output.LayerNorm.weight]
Loading weights:  67%|██████▋   | 263/391 [00:00<00:00, 3138.65it/s, Materializing param=encoder.layer.16.attention.output.LayerNorm.weight]
Loading weights:  68%|██████▊   | 264/391 [00:00<00:00, 3142.46it/s, Materializing param=encoder.layer.16.attention.output.dense.bias]      
Loading weights:  68%|██████▊   | 264/391 [00:00<00:00, 3137.53it/s, Materializing param=encoder.layer.16.attention.output.dense.bias]
Loading weights:  68%|██████▊   | 265/391 [00:00<00:00, 3142.05it/s, Materializing param=encoder.layer.16.attention.output.dense.weight]
Loading weights:  68%|██████▊   | 265/391 [00:00<00:00, 3137.46it/s, Materializing param=encoder.layer.16.attention.output.dense.weight]
Loading weights:  68%|██████▊   | 266/391 [00:00<00:00, 3142.14it/s, Materializing param=encoder.layer.16.attention.self.key.bias]      
Loading weights:  68%|██████▊   | 266/391 [00:00<00:00, 3137.67it/s, Materializing param=encoder.layer.16.attention.self.key.bias]
Loading weights:  68%|██████▊   | 267/391 [00:00<00:00, 3141.69it/s, Materializing param=encoder.layer.16.attention.self.key.weight]
Loading weights:  68%|██████▊   | 267/391 [00:00<00:00, 3136.83it/s, Materializing param=encoder.layer.16.attention.self.key.weight]
Loading weights:  69%|██████▊   | 268/391 [00:00<00:00, 3141.37it/s, Materializing param=encoder.layer.16.attention.self.query.bias]
Loading weights:  69%|██████▊   | 268/391 [00:00<00:00, 3136.96it/s, Materializing param=encoder.layer.16.attention.self.query.bias]
Loading weights:  69%|██████▉   | 269/391 [00:00<00:00, 3141.53it/s, Materializing param=encoder.layer.16.attention.self.query.weight]
Loading weights:  69%|██████▉   | 269/391 [00:00<00:00, 3137.14it/s, Materializing param=encoder.layer.16.attention.self.query.weight]
Loading weights:  69%|██████▉   | 270/391 [00:00<00:00, 3141.30it/s, Materializing param=encoder.layer.16.attention.self.value.bias]  
Loading weights:  69%|██████▉   | 270/391 [00:00<00:00, 3136.94it/s, Materializing param=encoder.layer.16.attention.self.value.bias]
Loading weights:  69%|██████▉   | 271/391 [00:00<00:00, 3142.04it/s, Materializing param=encoder.layer.16.attention.self.value.weight]
Loading weights:  69%|██████▉   | 271/391 [00:00<00:00, 3138.10it/s, Materializing param=encoder.layer.16.attention.self.value.weight]
Loading weights:  70%|██████▉   | 272/391 [00:00<00:00, 3143.45it/s, Materializing param=encoder.layer.16.intermediate.dense.bias]    
Loading weights:  70%|██████▉   | 272/391 [00:00<00:00, 3139.53it/s, Materializing param=encoder.layer.16.intermediate.dense.bias]
Loading weights:  70%|██████▉   | 273/391 [00:00<00:00, 3145.05it/s, Materializing param=encoder.layer.16.intermediate.dense.weight]
Loading weights:  70%|██████▉   | 273/391 [00:00<00:00, 3140.37it/s, Materializing param=encoder.layer.16.intermediate.dense.weight]
Loading weights:  70%|███████   | 274/391 [00:00<00:00, 3144.37it/s, Materializing param=encoder.layer.16.output.LayerNorm.bias]    
Loading weights:  70%|███████   | 274/391 [00:00<00:00, 3139.96it/s, Materializing param=encoder.layer.16.output.LayerNorm.bias]
Loading weights:  70%|███████   | 275/391 [00:00<00:00, 3144.42it/s, Materializing param=encoder.layer.16.output.LayerNorm.weight]
Loading weights:  70%|███████   | 275/391 [00:00<00:00, 3140.17it/s, Materializing param=encoder.layer.16.output.LayerNorm.weight]
Loading weights:  71%|███████   | 276/391 [00:00<00:00, 3144.81it/s, Materializing param=encoder.layer.16.output.dense.bias]      
Loading weights:  71%|███████   | 276/391 [00:00<00:00, 3140.22it/s, Materializing param=encoder.layer.16.output.dense.bias]
Loading weights:  71%|███████   | 277/391 [00:00<00:00, 3144.22it/s, Materializing param=encoder.layer.16.output.dense.weight]
Loading weights:  71%|███████   | 277/391 [00:00<00:00, 3139.79it/s, Materializing param=encoder.layer.16.output.dense.weight]
Loading weights:  71%|███████   | 278/391 [00:00<00:00, 3144.27it/s, Materializing param=encoder.layer.17.attention.output.LayerNorm.bias]
Loading weights:  71%|███████   | 278/391 [00:00<00:00, 3139.85it/s, Materializing param=encoder.layer.17.attention.output.LayerNorm.bias]
Loading weights:  71%|███████▏  | 279/391 [00:00<00:00, 3144.08it/s, Materializing param=encoder.layer.17.attention.output.LayerNorm.weight]
Loading weights:  71%|███████▏  | 279/391 [00:00<00:00, 3139.32it/s, Materializing param=encoder.layer.17.attention.output.LayerNorm.weight]
Loading weights:  72%|███████▏  | 280/391 [00:00<00:00, 3143.07it/s, Materializing param=encoder.layer.17.attention.output.dense.bias]      
Loading weights:  72%|███████▏  | 280/391 [00:00<00:00, 3138.77it/s, Materializing param=encoder.layer.17.attention.output.dense.bias]
Loading weights:  72%|███████▏  | 281/391 [00:00<00:00, 3143.03it/s, Materializing param=encoder.layer.17.attention.output.dense.weight]
Loading weights:  72%|███████▏  | 281/391 [00:00<00:00, 3138.79it/s, Materializing param=encoder.layer.17.attention.output.dense.weight]
Loading weights:  72%|███████▏  | 282/391 [00:00<00:00, 3143.11it/s, Materializing param=encoder.layer.17.attention.self.key.bias]      
Loading weights:  72%|███████▏  | 282/391 [00:00<00:00, 3138.44it/s, Materializing param=encoder.layer.17.attention.self.key.bias]
Loading weights:  72%|███████▏  | 283/391 [00:00<00:00, 3142.42it/s, Materializing param=encoder.layer.17.attention.self.key.weight]
Loading weights:  72%|███████▏  | 283/391 [00:00<00:00, 3137.58it/s, Materializing param=encoder.layer.17.attention.self.key.weight]
Loading weights:  73%|███████▎  | 284/391 [00:00<00:00, 3141.86it/s, Materializing param=encoder.layer.17.attention.self.query.bias]
Loading weights:  73%|███████▎  | 284/391 [00:00<00:00, 3137.87it/s, Materializing param=encoder.layer.17.attention.self.query.bias]
Loading weights:  73%|███████▎  | 285/391 [00:00<00:00, 3142.87it/s, Materializing param=encoder.layer.17.attention.self.query.weight]
Loading weights:  73%|███████▎  | 285/391 [00:00<00:00, 3139.14it/s, Materializing param=encoder.layer.17.attention.self.query.weight]
Loading weights:  73%|███████▎  | 286/391 [00:00<00:00, 3143.36it/s, Materializing param=encoder.layer.17.attention.self.value.bias]  
Loading weights:  73%|███████▎  | 286/391 [00:00<00:00, 3139.28it/s, Materializing param=encoder.layer.17.attention.self.value.bias]
Loading weights:  73%|███████▎  | 287/391 [00:00<00:00, 3144.19it/s, Materializing param=encoder.layer.17.attention.self.value.weight]
Loading weights:  73%|███████▎  | 287/391 [00:00<00:00, 3140.43it/s, Materializing param=encoder.layer.17.attention.self.value.weight]
Loading weights:  74%|███████▎  | 288/391 [00:00<00:00, 3145.49it/s, Materializing param=encoder.layer.17.intermediate.dense.bias]    
Loading weights:  74%|███████▎  | 288/391 [00:00<00:00, 3140.19it/s, Materializing param=encoder.layer.17.intermediate.dense.bias]
Loading weights:  74%|███████▍  | 289/391 [00:00<00:00, 3143.40it/s, Materializing param=encoder.layer.17.intermediate.dense.weight]
Loading weights:  74%|███████▍  | 289/391 [00:00<00:00, 3138.03it/s, Materializing param=encoder.layer.17.intermediate.dense.weight]
Loading weights:  74%|███████▍  | 290/391 [00:00<00:00, 3141.10it/s, Materializing param=encoder.layer.17.output.LayerNorm.bias]    
Loading weights:  74%|███████▍  | 290/391 [00:00<00:00, 3136.27it/s, Materializing param=encoder.layer.17.output.LayerNorm.bias]
Loading weights:  74%|███████▍  | 291/391 [00:00<00:00, 3139.55it/s, Materializing param=encoder.layer.17.output.LayerNorm.weight]
Loading weights:  74%|███████▍  | 291/391 [00:00<00:00, 3134.69it/s, Materializing param=encoder.layer.17.output.LayerNorm.weight]
Loading weights:  75%|███████▍  | 292/391 [00:00<00:00, 3137.03it/s, Materializing param=encoder.layer.17.output.dense.bias]      
Loading weights:  75%|███████▍  | 292/391 [00:00<00:00, 3132.87it/s, Materializing param=encoder.layer.17.output.dense.bias]
Loading weights:  75%|███████▍  | 293/391 [00:00<00:00, 3137.18it/s, Materializing param=encoder.layer.17.output.dense.weight]
Loading weights:  75%|███████▍  | 293/391 [00:00<00:00, 3133.16it/s, Materializing param=encoder.layer.17.output.dense.weight]
Loading weights:  75%|███████▌  | 294/391 [00:00<00:00, 3136.81it/s, Materializing param=encoder.layer.18.attention.output.LayerNorm.bias]
Loading weights:  75%|███████▌  | 294/391 [00:00<00:00, 3131.99it/s, Materializing param=encoder.layer.18.attention.output.LayerNorm.bias]
Loading weights:  75%|███████▌  | 295/391 [00:00<00:00, 3135.14it/s, Materializing param=encoder.layer.18.attention.output.LayerNorm.weight]
Loading weights:  75%|███████▌  | 295/391 [00:00<00:00, 3130.56it/s, Materializing param=encoder.layer.18.attention.output.LayerNorm.weight]
Loading weights:  76%|███████▌  | 296/391 [00:00<00:00, 3133.81it/s, Materializing param=encoder.layer.18.attention.output.dense.bias]      
Loading weights:  76%|███████▌  | 296/391 [00:00<00:00, 3129.22it/s, Materializing param=encoder.layer.18.attention.output.dense.bias]
Loading weights:  76%|███████▌  | 297/391 [00:00<00:00, 3133.40it/s, Materializing param=encoder.layer.18.attention.output.dense.weight]
Loading weights:  76%|███████▌  | 297/391 [00:00<00:00, 3128.94it/s, Materializing param=encoder.layer.18.attention.output.dense.weight]
Loading weights:  76%|███████▌  | 298/391 [00:00<00:00, 3132.51it/s, Materializing param=encoder.layer.18.attention.self.key.bias]      
Loading weights:  76%|███████▌  | 298/391 [00:00<00:00, 3128.43it/s, Materializing param=encoder.layer.18.attention.self.key.bias]
Loading weights:  76%|███████▋  | 299/391 [00:00<00:00, 3132.54it/s, Materializing param=encoder.layer.18.attention.self.key.weight]
Loading weights:  76%|███████▋  | 299/391 [00:00<00:00, 3128.57it/s, Materializing param=encoder.layer.18.attention.self.key.weight]
Loading weights:  77%|███████▋  | 300/391 [00:00<00:00, 3132.77it/s, Materializing param=encoder.layer.18.attention.self.query.bias]
Loading weights:  77%|███████▋  | 300/391 [00:00<00:00, 3128.53it/s, Materializing param=encoder.layer.18.attention.self.query.bias]
Loading weights:  77%|███████▋  | 301/391 [00:00<00:00, 3132.17it/s, Materializing param=encoder.layer.18.attention.self.query.weight]
Loading weights:  77%|███████▋  | 301/391 [00:00<00:00, 3128.20it/s, Materializing param=encoder.layer.18.attention.self.query.weight]
Loading weights:  77%|███████▋  | 302/391 [00:00<00:00, 3132.24it/s, Materializing param=encoder.layer.18.attention.self.value.bias]  
Loading weights:  77%|███████▋  | 302/391 [00:00<00:00, 3128.34it/s, Materializing param=encoder.layer.18.attention.self.value.bias]
Loading weights:  77%|███████▋  | 303/391 [00:00<00:00, 3132.45it/s, Materializing param=encoder.layer.18.attention.self.value.weight]
Loading weights:  77%|███████▋  | 303/391 [00:00<00:00, 3128.14it/s, Materializing param=encoder.layer.18.attention.self.value.weight]
Loading weights:  78%|███████▊  | 304/391 [00:00<00:00, 3132.12it/s, Materializing param=encoder.layer.18.intermediate.dense.bias]    
Loading weights:  78%|███████▊  | 304/391 [00:00<00:00, 3128.20it/s, Materializing param=encoder.layer.18.intermediate.dense.bias]
Loading weights:  78%|███████▊  | 305/391 [00:00<00:00, 3132.86it/s, Materializing param=encoder.layer.18.intermediate.dense.weight]
Loading weights:  78%|███████▊  | 305/391 [00:00<00:00, 3129.47it/s, Materializing param=encoder.layer.18.intermediate.dense.weight]
Loading weights:  78%|███████▊  | 306/391 [00:00<00:00, 3134.36it/s, Materializing param=encoder.layer.18.output.LayerNorm.bias]    
Loading weights:  78%|███████▊  | 306/391 [00:00<00:00, 3130.92it/s, Materializing param=encoder.layer.18.output.LayerNorm.bias]
Loading weights:  79%|███████▊  | 307/391 [00:00<00:00, 3134.97it/s, Materializing param=encoder.layer.18.output.LayerNorm.weight]
Loading weights:  79%|███████▊  | 307/391 [00:00<00:00, 3131.36it/s, Materializing param=encoder.layer.18.output.LayerNorm.weight]
Loading weights:  79%|███████▉  | 308/391 [00:00<00:00, 3135.87it/s, Materializing param=encoder.layer.18.output.dense.bias]      
Loading weights:  79%|███████▉  | 308/391 [00:00<00:00, 3132.49it/s, Materializing param=encoder.layer.18.output.dense.bias]
Loading weights:  79%|███████▉  | 309/391 [00:00<00:00, 3137.40it/s, Materializing param=encoder.layer.18.output.dense.weight]
Loading weights:  79%|███████▉  | 309/391 [00:00<00:00, 3134.03it/s, Materializing param=encoder.layer.18.output.dense.weight]
Loading weights:  79%|███████▉  | 310/391 [00:00<00:00, 3138.93it/s, Materializing param=encoder.layer.19.attention.output.LayerNorm.bias]
Loading weights:  79%|███████▉  | 310/391 [00:00<00:00, 3134.77it/s, Materializing param=encoder.layer.19.attention.output.LayerNorm.bias]
Loading weights:  80%|███████▉  | 311/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.LayerNorm.bias]
Loading weights:  80%|███████▉  | 311/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.LayerNorm.weight]
Loading weights:  80%|███████▉  | 311/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.LayerNorm.weight]
Loading weights:  80%|███████▉  | 312/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.dense.bias]      
Loading weights:  80%|███████▉  | 312/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.dense.bias]
Loading weights:  80%|████████  | 313/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.dense.weight]
Loading weights:  80%|████████  | 313/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.output.dense.weight]
Loading weights:  80%|████████  | 314/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.key.bias]      
Loading weights:  80%|████████  | 314/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.key.bias]
Loading weights:  81%|████████  | 315/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.key.weight]
Loading weights:  81%|████████  | 315/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.key.weight]
Loading weights:  81%|████████  | 316/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.query.bias]
Loading weights:  81%|████████  | 316/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.query.bias]
Loading weights:  81%|████████  | 317/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.query.weight]
Loading weights:  81%|████████  | 317/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.query.weight]
Loading weights:  81%|████████▏ | 318/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.value.bias]  
Loading weights:  81%|████████▏ | 318/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.value.bias]
Loading weights:  82%|████████▏ | 319/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.value.weight]
Loading weights:  82%|████████▏ | 319/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.attention.self.value.weight]
Loading weights:  82%|████████▏ | 320/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.intermediate.dense.bias]    
Loading weights:  82%|████████▏ | 320/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.intermediate.dense.bias]
Loading weights:  82%|████████▏ | 321/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.intermediate.dense.weight]
Loading weights:  82%|████████▏ | 321/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.intermediate.dense.weight]
Loading weights:  82%|████████▏ | 322/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.LayerNorm.bias]    
Loading weights:  82%|████████▏ | 322/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.LayerNorm.bias]
Loading weights:  83%|████████▎ | 323/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.LayerNorm.weight]
Loading weights:  83%|████████▎ | 323/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.LayerNorm.weight]
Loading weights:  83%|████████▎ | 324/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.dense.bias]      
Loading weights:  83%|████████▎ | 324/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.dense.bias]
Loading weights:  83%|████████▎ | 325/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.dense.weight]
Loading weights:  83%|████████▎ | 325/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.19.output.dense.weight]
Loading weights:  83%|████████▎ | 326/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.LayerNorm.bias]
Loading weights:  83%|████████▎ | 326/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.LayerNorm.bias]
Loading weights:  84%|████████▎ | 327/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.LayerNorm.weight]
Loading weights:  84%|████████▎ | 327/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.LayerNorm.weight]
Loading weights:  84%|████████▍ | 328/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.dense.bias]      
Loading weights:  84%|████████▍ | 328/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.dense.bias]
Loading weights:  84%|████████▍ | 329/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.dense.weight]
Loading weights:  84%|████████▍ | 329/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.output.dense.weight]
Loading weights:  84%|████████▍ | 330/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.key.bias]      
Loading weights:  84%|████████▍ | 330/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.key.bias]
Loading weights:  85%|████████▍ | 331/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.key.weight]
Loading weights:  85%|████████▍ | 331/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.key.weight]
Loading weights:  85%|████████▍ | 332/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.query.bias]
Loading weights:  85%|████████▍ | 332/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.query.bias]
Loading weights:  85%|████████▌ | 333/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.query.weight]
Loading weights:  85%|████████▌ | 333/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.query.weight]
Loading weights:  85%|████████▌ | 334/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.value.bias]  
Loading weights:  85%|████████▌ | 334/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.value.bias]
Loading weights:  86%|████████▌ | 335/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.value.weight]
Loading weights:  86%|████████▌ | 335/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.attention.self.value.weight]
Loading weights:  86%|████████▌ | 336/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.intermediate.dense.bias]    
Loading weights:  86%|████████▌ | 336/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.intermediate.dense.bias]
Loading weights:  86%|████████▌ | 337/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.intermediate.dense.weight]
Loading weights:  86%|████████▌ | 337/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.intermediate.dense.weight]
Loading weights:  86%|████████▋ | 338/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.LayerNorm.bias]    
Loading weights:  86%|████████▋ | 338/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.LayerNorm.bias]
Loading weights:  87%|████████▋ | 339/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.LayerNorm.weight]
Loading weights:  87%|████████▋ | 339/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.LayerNorm.weight]
Loading weights:  87%|████████▋ | 340/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.dense.bias]      
Loading weights:  87%|████████▋ | 340/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.dense.bias]
Loading weights:  87%|████████▋ | 341/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.dense.weight]
Loading weights:  87%|████████▋ | 341/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.20.output.dense.weight]
Loading weights:  87%|████████▋ | 342/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.LayerNorm.bias]
Loading weights:  87%|████████▋ | 342/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.LayerNorm.bias]
Loading weights:  88%|████████▊ | 343/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.LayerNorm.weight]
Loading weights:  88%|████████▊ | 343/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.LayerNorm.weight]
Loading weights:  88%|████████▊ | 344/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.dense.bias]      
Loading weights:  88%|████████▊ | 344/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.dense.bias]
Loading weights:  88%|████████▊ | 345/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.dense.weight]
Loading weights:  88%|████████▊ | 345/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.output.dense.weight]
Loading weights:  88%|████████▊ | 346/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.key.bias]      
Loading weights:  88%|████████▊ | 346/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.key.bias]
Loading weights:  89%|████████▊ | 347/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.key.weight]
Loading weights:  89%|████████▊ | 347/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.key.weight]
Loading weights:  89%|████████▉ | 348/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.query.bias]
Loading weights:  89%|████████▉ | 348/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.query.bias]
Loading weights:  89%|████████▉ | 349/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.query.weight]
Loading weights:  89%|████████▉ | 349/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.query.weight]
Loading weights:  90%|████████▉ | 350/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.value.bias]  
Loading weights:  90%|████████▉ | 350/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.value.bias]
Loading weights:  90%|████████▉ | 351/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.value.weight]
Loading weights:  90%|████████▉ | 351/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.attention.self.value.weight]
Loading weights:  90%|█████████ | 352/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.intermediate.dense.bias]    
Loading weights:  90%|█████████ | 352/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.intermediate.dense.bias]
Loading weights:  90%|█████████ | 353/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.intermediate.dense.weight]
Loading weights:  90%|█████████ | 353/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.intermediate.dense.weight]
Loading weights:  91%|█████████ | 354/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.LayerNorm.bias]    
Loading weights:  91%|█████████ | 354/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.LayerNorm.bias]
Loading weights:  91%|█████████ | 355/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.LayerNorm.weight]
Loading weights:  91%|█████████ | 355/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.LayerNorm.weight]
Loading weights:  91%|█████████ | 356/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.dense.bias]      
Loading weights:  91%|█████████ | 356/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.dense.bias]
Loading weights:  91%|█████████▏| 357/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.dense.weight]
Loading weights:  91%|█████████▏| 357/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.21.output.dense.weight]
Loading weights:  92%|█████████▏| 358/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.LayerNorm.bias]
Loading weights:  92%|█████████▏| 358/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.LayerNorm.bias]
Loading weights:  92%|█████████▏| 359/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.LayerNorm.weight]
Loading weights:  92%|█████████▏| 359/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.LayerNorm.weight]
Loading weights:  92%|█████████▏| 360/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.dense.bias]      
Loading weights:  92%|█████████▏| 360/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.dense.bias]
Loading weights:  92%|█████████▏| 361/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.dense.weight]
Loading weights:  92%|█████████▏| 361/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.output.dense.weight]
Loading weights:  93%|█████████▎| 362/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.key.bias]      
Loading weights:  93%|█████████▎| 362/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.key.bias]
Loading weights:  93%|█████████▎| 363/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.key.weight]
Loading weights:  93%|█████████▎| 363/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.key.weight]
Loading weights:  93%|█████████▎| 364/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.query.bias]
Loading weights:  93%|█████████▎| 364/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.query.bias]
Loading weights:  93%|█████████▎| 365/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.query.weight]
Loading weights:  93%|█████████▎| 365/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.query.weight]
Loading weights:  94%|█████████▎| 366/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.value.bias]  
Loading weights:  94%|█████████▎| 366/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.value.bias]
Loading weights:  94%|█████████▍| 367/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.value.weight]
Loading weights:  94%|█████████▍| 367/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.attention.self.value.weight]
Loading weights:  94%|█████████▍| 368/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.intermediate.dense.bias]    
Loading weights:  94%|█████████▍| 368/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.intermediate.dense.bias]
Loading weights:  94%|█████████▍| 369/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.intermediate.dense.weight]
Loading weights:  94%|█████████▍| 369/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.intermediate.dense.weight]
Loading weights:  95%|█████████▍| 370/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.LayerNorm.bias]    
Loading weights:  95%|█████████▍| 370/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.LayerNorm.bias]
Loading weights:  95%|█████████▍| 371/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.LayerNorm.weight]
Loading weights:  95%|█████████▍| 371/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.LayerNorm.weight]
Loading weights:  95%|█████████▌| 372/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.dense.bias]      
Loading weights:  95%|█████████▌| 372/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.dense.bias]
Loading weights:  95%|█████████▌| 373/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.dense.weight]
Loading weights:  95%|█████████▌| 373/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.22.output.dense.weight]
Loading weights:  96%|█████████▌| 374/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.LayerNorm.bias]
Loading weights:  96%|█████████▌| 374/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.LayerNorm.bias]
Loading weights:  96%|█████████▌| 375/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.LayerNorm.weight]
Loading weights:  96%|█████████▌| 375/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.LayerNorm.weight]
Loading weights:  96%|█████████▌| 376/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.dense.bias]      
Loading weights:  96%|█████████▌| 376/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.dense.bias]
Loading weights:  96%|█████████▋| 377/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.dense.weight]
Loading weights:  96%|█████████▋| 377/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.output.dense.weight]
Loading weights:  97%|█████████▋| 378/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.key.bias]      
Loading weights:  97%|█████████▋| 378/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.key.bias]
Loading weights:  97%|█████████▋| 379/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.key.weight]
Loading weights:  97%|█████████▋| 379/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.key.weight]
Loading weights:  97%|█████████▋| 380/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.query.bias]
Loading weights:  97%|█████████▋| 380/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.query.bias]
Loading weights:  97%|█████████▋| 381/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.query.weight]
Loading weights:  97%|█████████▋| 381/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.query.weight]
Loading weights:  98%|█████████▊| 382/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.value.bias]  
Loading weights:  98%|█████████▊| 382/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.value.bias]
Loading weights:  98%|█████████▊| 383/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.value.weight]
Loading weights:  98%|█████████▊| 383/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.attention.self.value.weight]
Loading weights:  98%|█████████▊| 384/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.intermediate.dense.bias]    
Loading weights:  98%|█████████▊| 384/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.intermediate.dense.bias]
Loading weights:  98%|█████████▊| 385/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.intermediate.dense.weight]
Loading weights:  98%|█████████▊| 385/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.intermediate.dense.weight]
Loading weights:  99%|█████████▊| 386/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.LayerNorm.bias]    
Loading weights:  99%|█████████▊| 386/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.LayerNorm.bias]
Loading weights:  99%|█████████▉| 387/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.LayerNorm.weight]
Loading weights:  99%|█████████▉| 387/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.LayerNorm.weight]
Loading weights:  99%|█████████▉| 388/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.dense.bias]      
Loading weights:  99%|█████████▉| 388/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.dense.bias]
Loading weights:  99%|█████████▉| 389/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.dense.weight]
Loading weights:  99%|█████████▉| 389/391 [00:00<00:00, 3107.24it/s, Materializing param=encoder.layer.23.output.dense.weight]
Loading weights: 100%|█████████▉| 390/391 [00:00<00:00, 3107.24it/s, Materializing param=pooler.dense.bias]                   
Loading weights: 100%|█████████▉| 390/391 [00:00<00:00, 3107.24it/s, Materializing param=pooler.dense.bias]
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 3107.24it/s, Materializing param=pooler.dense.weight]
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 3107.24it/s, Materializing param=pooler.dense.weight]
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 3017.81it/s, Materializing param=pooler.dense.weight]

Batches:   0%|          | 0/19 [00:00<?, ?it/s]
Batches:   5%|▌         | 1/19 [00:00<00:07,  2.46it/s]
Batches:  11%|█         | 2/19 [00:00<00:04,  3.64it/s]
Batches:  16%|█▌        | 3/19 [00:00<00:03,  4.29it/s]
Batches:  21%|██        | 4/19 [00:00<00:03,  4.67it/s]
Batches:  26%|██▋       | 5/19 [00:01<00:02,  4.88it/s]
Batches:  32%|███▏      | 6/19 [00:01<00:02,  5.26it/s]
Batches:  37%|███▋      | 7/19 [00:01<00:02,  5.48it/s]
Batches:  42%|████▏     | 8/19 [00:01<00:01,  5.67it/s]
Batches:  47%|████▋     | 9/19 [00:01<00:01,  5.85it/s]
Batches:  53%|█████▎    | 10/19 [00:01<00:01,  6.15it/s]
Batches:  58%|█████▊    | 11/19 [00:02<00:01,  6.61it/s]
Batches:  63%|██████▎   | 12/19 [00:02<00:01,  6.96it/s]
Batches:  68%|██████▊   | 13/19 [00:02<00:00,  7.23it/s]
Batches:  74%|███████▎  | 14/19 [00:02<00:00,  7.40it/s]
Batches:  79%|███████▉  | 15/19 [00:02<00:00,  7.57it/s]
Batches:  84%|████████▍ | 16/19 [00:02<00:00,  7.70it/s]
Batches:  95%|█████████▍| 18/19 [00:02<00:00,  9.74it/s]
Batches: 100%|██████████| 19/19 [00:02<00:00,  6.59it/s]
/content/husc-admission-chat-enrollment/rag2025/scripts/ingest_lancedb.py:173: DeprecationWarning: table_names() is deprecated, use list_tables() instead
  table_exists = settings.LANCEDB_TABLE in db.table_names()
2026-04-18 18:03:17.287 | INFO     | __main__:ingest_lancedb:190 - Full ingest complete (overwrite): table=rag2025_bge, rows=152, uri=./data/lancedb


>>> Step 2: Build Knowledge Graph
2026-04-18 18:03:20.280 | INFO     | __main__:main:124 - === Graph Builder: Starting ===
2026-04-18 18:03:20.280 | INFO     | __main__:main:125 -   Mode: full | dry_run=False | limit=all
2026-04-18 18:03:20.287 | INFO     | __main__:load_all_chunks:78 - Loaded 152 chunks from 12 JSONL files
2026-04-18 18:03:20.287 | INFO     | src.services.llm_client:__init__:144 - UnifiedLLMClient initialized: providers=['ramclouds']
2026-04-18 18:03:20.288 | INFO     | __main__:main:168 - NER provider: ramclouds | model: gpt-5.4 | base_url: https://ramclouds.me/v1
2026-04-18 18:03:20.288 | INFO     | __main__:main:173 - Step 1: NER extraction (152 chunks via gpt-5.4)...
2026-04-18 18:03:20.288 | INFO     | src.services.ner_service:extract_batch:130 - NER: 1/152 – all_bogddt_tt_08_header_2025
2026-04-18 18:03:21.850 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d3abd40 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:23.018 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3ab530 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:23.018 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_bogddt_tt_08_header_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:23.018 | INFO     | src.services.ner_service:extract_batch:130 - NER: 2/152 – all_bogddt_tt_08_dieu_2_nguong_dau_vao_2025
2026-04-18 18:03:24.187 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca005c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:25.359 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9eab10 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:25.359 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_bogddt_tt_08_dieu_2_nguong_dau_vao_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:25.359 | INFO     | src.services.ner_service:extract_batch:130 - NER: 3/152 – csdt_admin_bogddt_tt_08_dieu_2_xu_ly_nguyen_vong_2025
2026-04-18 18:03:26.533 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca03e00 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:27.711 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca01910 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:27.712 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_bogddt_tt_08_dieu_2_xu_ly_nguyen_vong_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:27.712 | INFO     | src.services.ner_service:extract_batch:130 - NER: 4/152 – thi_sinh_bogddt_tt_08_dieu_2_diem_uu_tien_2025
2026-04-18 18:03:28.886 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca16cf0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:30.062 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14050 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:30.062 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_2_diem_uu_tien_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:30.063 | INFO     | src.services.ner_service:extract_batch:130 - NER: 5/152 – thi_sinh_bogddt_tt_08_dieu_2_xet_tuyen_thang_2025
2026-04-18 18:03:31.250 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca31910 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:32.474 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14cb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:32.474 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_2_xet_tuyen_thang_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:32.475 | INFO     | src.services.ner_service:extract_batch:130 - NER: 6/152 – thi_sinh_bogddt_tt_08_dieu_4_bao_ve_quyen_loi_2025
2026-04-18 18:03:33.690 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca48560 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:34.886 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca312e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:34.886 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_4_bao_ve_quyen_loi_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:34.887 | INFO     | src.services.ner_service:extract_batch:130 - NER: 7/152 – csdt_admin_bogddt_tt_08_dieu_6_gioi_han_to_hop_2025
2026-04-18 18:03:36.055 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca14b60 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:37.227 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca15e50 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:37.227 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_bogddt_tt_08_dieu_6_gioi_han_to_hop_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:37.227 | INFO     | src.services.ner_service:extract_batch:130 - NER: 8/152 – thi_sinh_bogddt_tt_08_dieu_7_muc_diem_uu_tien_kv_2025
2026-04-18 18:03:38.407 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca030e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:39.574 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca33aa0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:39.574 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_7_muc_diem_uu_tien_kv_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:39.574 | INFO     | src.services.ner_service:extract_batch:130 - NER: 9/152 – thi_sinh_bogddt_tt_08_dieu_7_cong_thuc_giam_uu_tien_2025
2026-04-18 18:03:40.744 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d3ab4d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:41.939 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14620 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:41.939 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_7_cong_thuc_giam_uu_tien_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:41.940 | INFO     | src.services.ner_service:extract_batch:130 - NER: 10/152 – thi_sinh_bogddt_tt_08_dieu_8_tuyen_thang_hsg_khkt_2025
2026-04-18 18:03:43.111 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d378ad0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:44.310 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca140e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:44.310 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_8_tuyen_thang_hsg_khkt_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:44.310 | INFO     | src.services.ner_service:extract_batch:130 - NER: 11/152 – thi_sinh_bogddt_tt_08_dieu_8_uu_tien_giai_khuyen_khich_2025
2026-04-18 18:03:45.508 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b440 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:46.679 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4be60 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:46.679 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_8_uu_tien_giai_khuyen_khich_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:46.679 | INFO     | src.services.ner_service:extract_batch:130 - NER: 12/152 – thi_sinh_bogddt_tt_08_dieu_9_nguong_dau_vao_pp_khac_2025
2026-04-18 18:03:47.845 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca0a300 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:49.017 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4b350 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:49.018 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_9_nguong_dau_vao_pp_khac_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:49.018 | INFO     | src.services.ner_service:extract_batch:130 - NER: 13/152 – thi_sinh_bogddt_tt_08_dieu_9_nguong_dau_vao_pp_khac_dac_thu_2025
2026-04-18 18:03:50.221 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca33f50 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:51.396 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca09190 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:51.396 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_9_nguong_dau_vao_pp_khac_dac_thu_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:51.397 | INFO     | src.services.ner_service:extract_batch:130 - NER: 14/152 – thi_sinh_bogddt_tt_08_dieu_9_mien_nguong_vdv_nang_khieu_2025
2026-04-18 18:03:52.569 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca57f50 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:53.743 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e9310 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:53.743 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_9_mien_nguong_vdv_nang_khieu_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:53.743 | INFO     | src.services.ner_service:extract_batch:130 - NER: 15/152 – thi_sinh_bogddt_tt_08_dieu_21_xac_nhan_nhap_hoc_online_2025
2026-04-18 18:03:54.905 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d3ab4d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:56.102 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e9fd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:56.103 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_21_xac_nhan_nhap_hoc_online_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:56.103 | INFO     | src.services.ner_service:extract_batch:130 - NER: 16/152 – thi_sinh_bogddt_tt_08_dieu_21_hau_qua_khong_xac_nhan_2025
2026-04-18 18:03:57.331 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca563f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:58.499 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca64e60 state=finished raised PermissionDeniedError>]
2026-04-18 18:03:58.499 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_21_hau_qua_khong_xac_nhan_2025: All LLM providers failed for JSON mode
2026-04-18 18:03:58.499 | INFO     | src.services.ner_service:extract_batch:130 - NER: 17/152 – thi_sinh_bogddt_tt_08_dieu_21_cam_xet_tuyen_sau_xac_nhan_2025
2026-04-18 18:03:59.666 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7a2d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:00.841 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca7a330 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:00.841 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_bogddt_tt_08_dieu_21_cam_xet_tuyen_sau_xac_nhan_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:00.841 | INFO     | src.services.ner_service:extract_batch:130 - NER: 18/152 – csdt_admin_bogddt_tt_08_dieu_26_cap_nhat_du_lieu_nhap_hoc_2025
2026-04-18 18:04:02.011 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8b920 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:03.206 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca899d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:03.206 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_bogddt_tt_08_dieu_26_cap_nhat_du_lieu_nhap_hoc_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:03.207 | INFO     | src.services.ner_service:extract_batch:130 - NER: 19/152 – husc_nganh_7480107TD_2025
2026-04-18 18:04:04.371 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca78c20 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:05.656 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca7b050 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:05.657 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7480107TD_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:05.657 | INFO     | src.services.ner_service:extract_batch:130 - NER: 20/152 – husc_nganh_7480201_2025
2026-04-18 18:04:06.836 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca67380 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:08.066 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca88620 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:08.067 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7480201_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:08.067 | INFO     | src.services.ner_service:extract_batch:130 - NER: 21/152 – husc_nganh_7480103_2025
2026-04-18 18:04:09.274 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca57d10 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:10.444 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca65070 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:10.444 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7480103_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:10.444 | INFO     | src.services.ner_service:extract_batch:130 - NER: 22/152 – husc_nganh_7480201VJ_2025
2026-04-18 18:04:11.616 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca54aa0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:12.788 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca54e00 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:12.789 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7480201VJ_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:12.789 | INFO     | src.services.ner_service:extract_batch:130 - NER: 23/152 – husc_nganh_7510302_2025
2026-04-18 18:04:14.149 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca78c50 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:15.331 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca797f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:15.332 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7510302_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:15.332 | INFO     | src.services.ner_service:extract_batch:130 - NER: 24/152 – husc_nganh_7440102_2025
2026-04-18 18:04:16.504 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4bfe0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:17.668 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b5f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:17.668 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7440102_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:17.668 | INFO     | src.services.ner_service:extract_batch:130 - NER: 25/152 – husc_nganh_7440112_2025
2026-04-18 18:04:18.877 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9ea120 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:20.157 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca49250 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:20.157 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7440112_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:20.158 | INFO     | src.services.ner_service:extract_batch:130 - NER: 26/152 – husc_nganh_7510401_2025
2026-04-18 18:04:21.365 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8bbf0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:22.557 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9eba10 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:22.557 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7510401_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:22.557 | INFO     | src.services.ner_service:extract_batch:130 - NER: 27/152 – husc_nganh_7420201_2025
2026-04-18 18:04:23.730 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca33530 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:24.902 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca00320 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:24.902 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7420201_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:24.902 | INFO     | src.services.ner_service:extract_batch:130 - NER: 28/152 – husc_nganh_7850101_2025
2026-04-18 18:04:26.071 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8be60 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:27.238 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14980 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:27.238 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7850101_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:27.238 | INFO     | src.services.ner_service:extract_batch:130 - NER: 29/152 – husc_nganh_7520503_2025
2026-04-18 18:04:28.410 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca990d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:29.592 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4bb60 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:29.593 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7520503_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:29.593 | INFO     | src.services.ner_service:extract_batch:130 - NER: 30/152 – husc_nganh_7580211_2025
2026-04-18 18:04:30.800 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca9ac90 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:32.085 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b41d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:32.085 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7580211_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:32.085 | INFO     | src.services.ner_service:extract_batch:130 - NER: 31/152 – husc_nganh_7440301_2025
2026-04-18 18:04:33.299 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca98da0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:34.480 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca9b590 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:34.480 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7440301_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:34.480 | INFO     | src.services.ner_service:extract_batch:130 - NER: 32/152 – husc_nganh_7850105_2025
2026-04-18 18:04:35.647 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca15010 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:36.822 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b7830 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:36.823 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7850105_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:36.823 | INFO     | src.services.ner_service:extract_batch:130 - NER: 33/152 – husc_nganh_7580101_2025
2026-04-18 18:04:37.981 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca03500 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:39.155 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca177d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:39.155 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7580101_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:39.155 | INFO     | src.services.ner_service:extract_batch:130 - NER: 34/152 – husc_nganh_7220104_2025
2026-04-18 18:04:40.318 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca16d50 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:41.494 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e95e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:41.494 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7220104_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:41.494 | INFO     | src.services.ner_service:extract_batch:130 - NER: 35/152 – husc_nganh_7229030_2025
2026-04-18 18:04:42.694 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca49490 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:43.946 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca02a20 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:43.946 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7229030_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:43.947 | INFO     | src.services.ner_service:extract_batch:130 - NER: 36/152 – husc_nganh_7229010_2025
2026-04-18 18:04:45.143 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca00140 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:46.328 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca499d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:46.328 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7229010_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:46.328 | INFO     | src.services.ner_service:extract_batch:130 - NER: 37/152 – husc_nganh_7310608_2025
2026-04-18 18:04:47.495 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca09eb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:48.668 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0bbc0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:48.668 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7310608_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:48.668 | INFO     | src.services.ner_service:extract_batch:130 - NER: 38/152 – husc_nganh_7229042_2025
2026-04-18 18:04:49.836 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca56de0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:51.032 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d44d040 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:51.032 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7229042_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:51.032 | INFO     | src.services.ner_service:extract_batch:130 - NER: 39/152 – husc_nganh_7229001_2025
2026-04-18 18:04:52.203 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca66690 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:53.375 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca563c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:53.375 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7229001_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:53.375 | INFO     | src.services.ner_service:extract_batch:130 - NER: 40/152 – husc_nganh_7310205_2025
2026-04-18 18:04:54.583 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca66b70 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:55.824 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0a450 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:55.824 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7310205_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:55.824 | INFO     | src.services.ner_service:extract_batch:130 - NER: 41/152 – husc_nganh_7320101_2025
2026-04-18 18:04:57.028 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7bc80 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:58.210 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca78b00 state=finished raised PermissionDeniedError>]
2026-04-18 18:04:58.210 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7320101_2025: All LLM providers failed for JSON mode
2026-04-18 18:04:58.211 | INFO     | src.services.ner_service:extract_batch:130 - NER: 42/152 – husc_nganh_7320111_2025
2026-04-18 18:04:59.383 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca65fa0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:00.590 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b9370 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:00.590 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7320111_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:00.591 | INFO     | src.services.ner_service:extract_batch:130 - NER: 43/152 – husc_nganh_7310301_2025
2026-04-18 18:05:01.773 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7a6f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:03.012 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b8620 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:03.012 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7310301_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:03.012 | INFO     | src.services.ner_service:extract_batch:130 - NER: 44/152 – husc_nganh_7760101_2025
2026-04-18 18:05:04.443 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b83e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:05.614 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca797f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:05.614 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_nganh_7760101_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:05.614 | INFO     | src.services.ner_service:extract_batch:130 - NER: 45/152 – husc_summary_general_2025
2026-04-18 18:05:06.810 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7a510 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:08.049 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca8ae10 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:08.050 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_summary_general_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:08.050 | INFO     | src.services.ner_service:extract_batch:130 - NER: 46/152 – husc_summary_tech_2025
2026-04-18 18:05:09.260 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b8bc0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:10.442 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca88b00 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:10.442 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_summary_tech_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:10.442 | INFO     | src.services.ner_service:extract_batch:130 - NER: 47/152 – husc_summary_social_2025
2026-04-18 18:05:11.618 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca67560 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:12.786 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d378ce0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:12.786 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_summary_social_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:12.786 | INFO     | src.services.ner_service:extract_batch:130 - NER: 48/152 – husc_summary_combos_C_2025
2026-04-18 18:05:13.959 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7ac00 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:15.127 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca557f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:15.127 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_summary_combos_C_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:15.127 | INFO     | src.services.ner_service:extract_batch:130 - NER: 49/152 – husc_hoc_phi_tin_chi_2025
2026-04-18 18:05:16.306 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca553a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:17.475 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b590 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:17.475 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_hoc_phi_tin_chi_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:17.476 | INFO     | src.services.ner_service:extract_batch:130 - NER: 50/152 – husc_uoc_tinh_hoc_phi_ky_2025
2026-04-18 18:05:18.682 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca55340 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:19.918 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3ab560 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:19.918 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_uoc_tinh_hoc_phi_ky_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:19.918 | INFO     | src.services.ner_service:extract_batch:130 - NER: 51/152 – husc_news_0001_diem_chuan_2025
2026-04-18 18:05:21.147 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca08f80 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:22.318 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca88050 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:22.318 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0001_diem_chuan_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:22.318 | INFO     | src.services.ner_service:extract_batch:130 - NER: 52/152 – husc_news_0001_thu_tuc_nhap_hoc
2026-04-18 18:05:23.484 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b500 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:24.654 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca54230 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:24.654 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0001_thu_tuc_nhap_hoc: All LLM providers failed for JSON mode
2026-04-18 18:05:24.654 | INFO     | src.services.ner_service:extract_batch:130 - NER: 53/152 – husc_news_0002_ho_so_nhap_hoc_chi_tiet
2026-04-18 18:05:25.835 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b320 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:27.001 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca01f70 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:27.002 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0002_ho_so_nhap_hoc_chi_tiet: All LLM providers failed for JSON mode
2026-04-18 18:05:27.002 | INFO     | src.services.ner_service:extract_batch:130 - NER: 54/152 – husc_news_0003_xet_tuyen_bo_sung_2025
2026-04-18 18:05:28.193 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9e8590 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:29.364 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d44ff20 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:29.364 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0003_xet_tuyen_bo_sung_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:29.365 | INFO     | src.services.ner_service:extract_batch:130 - NER: 55/152 – husc_news_0004_diem_chuan_chinh_thuc_2025
2026-04-18 18:05:30.600 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca03770 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:31.829 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e8350 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:31.830 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0004_diem_chuan_chinh_thuc_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:31.830 | INFO     | src.services.ner_service:extract_batch:130 - NER: 56/152 – husc_news_0006_xac_nhan_nhap_hoc_bo_gd
2026-04-18 18:05:33.061 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9ebe30 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:34.233 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca48710 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:34.234 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0006_xac_nhan_nhap_hoc_bo_gd: All LLM providers failed for JSON mode
2026-04-18 18:05:34.234 | INFO     | src.services.ner_service:extract_batch:130 - NER: 57/152 – husc_news_0007_dang_ky_ktx_online
2026-04-18 18:05:35.409 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca49af0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:36.572 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3aaa80 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:36.572 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0007_dang_ky_ktx_online: All LLM providers failed for JSON mode
2026-04-18 18:05:36.573 | INFO     | src.services.ner_service:extract_batch:130 - NER: 58/152 – husc_news_0010_hoc_bong_tuyen_sinh_2025
2026-04-18 18:05:37.751 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9ea150 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:38.922 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca484d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:38.922 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0010_hoc_bong_tuyen_sinh_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:38.922 | INFO     | src.services.ner_service:extract_batch:130 - NER: 59/152 – husc_news_0021_diem_uu_tien_2025
2026-04-18 18:05:40.142 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b7a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:41.322 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca541d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:41.322 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0021_diem_uu_tien_2025: All LLM providers failed for JSON mode
2026-04-18 18:05:41.323 | INFO     | src.services.ner_service:extract_batch:130 - NER: 60/152 – husc_news_0040_nganh_data_science
2026-04-18 18:05:42.533 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca0b200 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:43.768 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca650d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:43.768 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0040_nganh_data_science: All LLM providers failed for JSON mode
2026-04-18 18:05:43.769 | INFO     | src.services.ner_service:extract_batch:130 - NER: 61/152 – husc_news_0041_nganh_cntt
2026-04-18 18:05:44.977 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca55460 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:46.144 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b7d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:46.145 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0041_nganh_cntt: All LLM providers failed for JSON mode
2026-04-18 18:05:46.145 | INFO     | src.services.ner_service:extract_batch:130 - NER: 62/152 – husc_news_0072_vi_sao_chon_cntt
2026-04-18 18:05:47.329 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca659d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:48.493 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca7b5c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:48.493 | WARNING  | src.services.ner_service:extract:89 - NER failed for husc_news_0072_vi_sao_chon_cntt: All LLM providers failed for JSON mode
2026-04-18 18:05:48.493 | INFO     | src.services.ner_service:extract_batch:130 - NER: 63/152 – all_Bo_GDDT_ma_phuong_thuc_xet_tuyen_2025_chunk_0
2026-04-18 18:05:49.687 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0bb3e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:50.861 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca56210 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:50.862 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_Bo_GDDT_ma_phuong_thuc_xet_tuyen_2025_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:05:50.862 | INFO     | src.services.ner_service:extract_batch:130 - NER: 64/152 – csdt_admin_Bo_GDDT_ma_to_hop_xet_tuyen_2025_chunk_0
2026-04-18 18:05:52.029 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca78c80 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:53.199 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca568d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:53.199 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_Bo_GDDT_ma_to_hop_xet_tuyen_2025_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:05:53.199 | INFO     | src.services.ner_service:extract_batch:130 - NER: 65/152 – thi_sinh_dai_hoc_hue_cac_phuong_thuc_tuyen_sinh_2025_chunk_0
2026-04-18 18:05:54.415 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8af30 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:55.692 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3790a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:55.692 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_cac_phuong_thuc_tuyen_sinh_2025_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:05:55.693 | INFO     | src.services.ner_service:extract_batch:130 - NER: 66/152 – thi_sinh_dai_hoc_hue_danh_muc_to_hop_nganh_2025_chunk_0
2026-04-18 18:05:56.897 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4a390 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:58.070 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14290 state=finished raised PermissionDeniedError>]
2026-04-18 18:05:58.071 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_danh_muc_to_hop_nganh_2025_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:05:58.071 | INFO     | src.services.ner_service:extract_batch:130 - NER: 67/152 – thi_sinh_dai_hoc_hue_thong_bao_tuyen_thang_2025
2026-04-18 18:05:59.270 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca15160 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:00.433 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca31cd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:00.433 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_thong_bao_tuyen_thang_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:00.433 | INFO     | src.services.ner_service:extract_batch:130 - NER: 68/152 – thi_sinh_dai_hoc_hue_tieu_chi_xet_tuyen_phu_2025
2026-04-18 18:06:01.599 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca32a20 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:02.777 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca154f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:02.777 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tieu_chi_xet_tuyen_phu_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:02.778 | INFO     | src.services.ner_service:extract_batch:130 - NER: 69/152 – thi_sinh_dai_hoc_hue_ty_le_chi_tieu_chung_2025
2026-04-18 18:06:03.957 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca14290 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:05.125 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca79c40 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:05.126 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_ty_le_chi_tieu_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:05.126 | INFO     | src.services.ner_service:extract_batch:130 - NER: 70/152 – thi_sinh_y_duoc_chi_tieu_db_cu_tuyen_2025
2026-04-18 18:06:06.334 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca88ef0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:07.562 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0bb950 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:07.563 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_y_duoc_chi_tieu_db_cu_tuyen_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:07.563 | INFO     | src.services.ner_service:extract_batch:130 - NER: 71/152 – thi_sinh_su_pham_chi_tieu_du_bi_2025
2026-04-18 18:06:08.768 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8a630 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:09.944 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca64e00 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:09.945 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_su_pham_chi_tieu_du_bi_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:09.945 | INFO     | src.services.ner_service:extract_batch:130 - NER: 72/152 – thi_sinh_dai_hoc_hue_tuyen_thang_anh_hung_2025
2026-04-18 18:06:11.119 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca64920 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:12.294 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca66a50 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:12.295 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_anh_hung_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:12.295 | INFO     | src.services.ner_service:extract_batch:130 - NER: 73/152 – thi_sinh_dai_hoc_hue_tuyen_thang_hsg_khkt_chung_2025
2026-04-18 18:06:13.487 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d378ad0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:14.661 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca88d70 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:14.662 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_hsg_khkt_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:14.662 | INFO     | src.services.ner_service:extract_batch:130 - NER: 74/152 – thi_sinh_y_duoc_tuyen_thang_khkt_2025
2026-04-18 18:06:15.840 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca575c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:17.014 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca54170 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:17.014 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_y_duoc_tuyen_thang_khkt_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:17.015 | INFO     | src.services.ner_service:extract_batch:130 - NER: 75/152 – thi_sinh_khoa_hoc_tuyen_thang_khkt_2025
2026-04-18 18:06:18.212 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d3ab440 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:19.441 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9ea900 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:19.442 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoa_hoc_tuyen_thang_khkt_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:19.442 | INFO     | src.services.ner_service:extract_batch:130 - NER: 76/152 – thi_sinh_khoa_hoc_kien_truc_dieu_kien_ve_2025
2026-04-18 18:06:20.760 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca15be0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:21.931 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9eb590 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:21.931 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoa_hoc_kien_truc_dieu_kien_ve_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:21.932 | INFO     | src.services.ner_service:extract_batch:130 - NER: 77/152 – thi_sinh_dai_hoc_hue_tuyen_thang_nghe_thuat_2025
2026-04-18 18:06:23.098 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca49190 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:24.272 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca78080 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:24.273 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_nghe_thuat_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:24.273 | INFO     | src.services.ner_service:extract_batch:130 - NER: 78/152 – thi_sinh_dai_hoc_hue_tuyen_thang_giao_duc_the_chat_2025
2026-04-18 18:06:25.444 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca57da0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:26.603 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca9a150 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:26.603 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_giao_duc_the_chat_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:26.604 | INFO     | src.services.ner_service:extract_batch:130 - NER: 79/152 – thi_sinh_dai_hoc_hue_tuyen_thang_nguoi_khuyet_tat_2025
2026-04-18 18:06:27.811 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca9a060 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:28.980 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b7440 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:28.981 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_nguoi_khuyet_tat_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:28.981 | INFO     | src.services.ner_service:extract_batch:130 - NER: 80/152 – thi_sinh_dai_hoc_hue_tuyen_thang_nguoi_nuoc_ngoai_chung_2025
2026-04-18 18:06:30.187 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b5be0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:31.435 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4aba0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:31.435 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_tuyen_thang_nguoi_nuoc_ngoai_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:31.435 | INFO     | src.services.ner_service:extract_batch:130 - NER: 81/152 – thi_sinh_y_duoc_tuyen_thang_nguoi_nuoc_ngoai_2025
2026-04-18 18:06:32.647 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b6d20 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:33.818 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b4a10 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:33.818 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_y_duoc_tuyen_thang_nguoi_nuoc_ngoai_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:33.819 | INFO     | src.services.ner_service:extract_batch:130 - NER: 82/152 – thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_giai_kk_quyen_2025
2026-04-18 18:06:35.000 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4a990 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:36.168 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca083e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:36.169 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_giai_kk_quyen_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:36.169 | INFO     | src.services.ner_service:extract_batch:130 - NER: 83/152 – thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_the_thao_2025
2026-04-18 18:06:37.344 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9eb6b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:38.515 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca00920 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:38.515 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_the_thao_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:38.516 | INFO     | src.services.ner_service:extract_batch:130 - NER: 84/152 – thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_nghe_thuat_2025
2026-04-18 18:06:39.685 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca481a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:40.850 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca57260 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:40.850 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_uu_tien_xet_tuyen_nghe_thuat_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:40.851 | INFO     | src.services.ner_service:extract_batch:130 - NER: 85/152 – thi_sinh_y_duoc_uu_tien_du_bi_dai_hoc_2025
2026-04-18 18:06:42.059 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca56a20 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:43.290 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca644d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:43.290 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_y_duoc_uu_tien_du_bi_dai_hoc_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:43.290 | INFO     | src.services.ner_service:extract_batch:130 - NER: 86/152 – thi_sinh_su_pham_uu_tien_du_bi_dao_tao_giao_vien_2025
2026-04-18 18:06:44.496 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca65220 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:45.667 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca79a60 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:45.667 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_su_pham_uu_tien_du_bi_dao_tao_giao_vien_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:45.668 | INFO     | src.services.ner_service:extract_batch:130 - NER: 87/152 – thi_sinh_su_pham_uu_tien_du_bi_giao_duc_mam_non_2025
2026-04-18 18:06:46.855 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca56120 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:48.024 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca8aa80 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:48.025 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_su_pham_uu_tien_du_bi_giao_duc_mam_non_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:48.025 | INFO     | src.services.ner_service:extract_batch:130 - NER: 88/152 – thi_sinh_su_pham_uu_tien_du_bi_su_pham_am_nhac_2025
2026-04-18 18:06:49.195 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca16600 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:50.400 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b650 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:50.400 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_su_pham_uu_tien_du_bi_su_pham_am_nhac_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:50.401 | INFO     | src.services.ner_service:extract_batch:130 - NER: 89/152 – thi_sinh_dai_hoc_hue_dieu_kien_lop_12_giao_vien_2025
2026-04-18 18:06:51.564 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca8abd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:52.734 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca7b350 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:52.735 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_dieu_kien_lop_12_giao_vien_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:52.735 | INFO     | src.services.ner_service:extract_batch:130 - NER: 90/152 – thi_sinh_dai_hoc_hue_dieu_kien_lop_12_khac_2025
2026-04-18 18:06:53.956 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca33260 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:55.189 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0ccc20 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:55.190 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_dieu_kien_lop_12_khac_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:55.190 | INFO     | src.services.ner_service:extract_batch:130 - NER: 91/152 – thi_sinh_su_pham_dieu_kien_suc_khoe_hanh_kiem_2025
2026-04-18 18:06:56.437 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b8920 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:57.609 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0ce120 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:57.609 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_su_pham_dieu_kien_suc_khoe_hanh_kiem_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:57.610 | INFO     | src.services.ner_service:extract_batch:130 - NER: 92/152 – thi_sinh_dai_hoc_hue_dieu_kien_the_hinh_giao_duc_the_chat_2025
2026-04-18 18:06:58.779 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0ce690 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:59.956 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14500 state=finished raised PermissionDeniedError>]
2026-04-18 18:06:59.956 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_dieu_kien_the_hinh_giao_duc_the_chat_2025: All LLM providers failed for JSON mode
2026-04-18 18:06:59.956 | INFO     | src.services.ner_service:extract_batch:130 - NER: 93/152 – thi_sinh_dai_hoc_hue_ho_so_xet_tuyen_thang_chung_2025
2026-04-18 18:07:01.141 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca88e90 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:02.314 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0cca10 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:02.314 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_ho_so_xet_tuyen_thang_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:02.314 | INFO     | src.services.ner_service:extract_batch:130 - NER: 94/152 – thi_sinh_dai_hoc_hue_ho_so_nguoi_khuyet_tat_2025
2026-04-18 18:07:03.484 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca302c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:04.649 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca033b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:04.649 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_ho_so_nguoi_khuyet_tat_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:04.649 | INFO     | src.services.ner_service:extract_batch:130 - NER: 95/152 – thi_sinh_dai_hoc_hue_le_phi_xet_tuyen_2025
2026-04-18 18:07:05.855 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca16600 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:07.086 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca66f90 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:07.086 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_le_phi_xet_tuyen_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:07.086 | INFO     | src.services.ner_service:extract_batch:130 - NER: 96/152 – thi_sinh_dai_hoc_hue_dia_diem_va_hinh_thuc_nop_ho_so_2025
2026-04-18 18:07:08.304 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca57f50 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:09.467 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3783b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:09.467 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_dia_diem_va_hinh_thuc_nop_ho_so_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:09.467 | INFO     | src.services.ner_service:extract_batch:130 - NER: 97/152 – thi_sinh_dai_hoc_hue_han_chot_nop_ho_so_2025
2026-04-18 18:07:10.639 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca64a70 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:11.811 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca08f80 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:11.812 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_han_chot_nop_ho_so_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:11.812 | INFO     | src.services.ner_service:extract_batch:130 - NER: 98/152 – thi_sinh_dai_hoc_hue_thoi_gian_cong_bo_ket_qua_2025
2026-04-18 18:07:12.994 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca67590 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:14.160 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca8b080 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:14.160 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_thoi_gian_cong_bo_ket_qua_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:14.160 | INFO     | src.services.ner_service:extract_batch:130 - NER: 99/152 – thi_sinh_dai_hoc_hue_lien_he_ho_tro_2025
2026-04-18 18:07:15.337 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca33890 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:16.516 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b6d50 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:16.516 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_lien_he_ho_tro_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:16.516 | INFO     | src.services.ner_service:extract_batch:130 - NER: 100/152 – all_dai_hoc_hue_thong_tin_chung_2025
2026-04-18 18:07:17.723 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca98b60 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:18.961 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b6f30 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:18.961 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_dai_hoc_hue_thong_tin_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:18.961 | INFO     | src.services.ner_service:extract_batch:130 - NER: 101/152 – thi_sinh_dai_hoc_hue_cac_phuong_thuc_tuyen_sinh_2025
2026-04-18 18:07:20.180 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b5610 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:21.351 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4b890 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:21.351 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_dai_hoc_hue_cac_phuong_thuc_tuyen_sinh_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:21.351 | INFO     | src.services.ner_service:extract_batch:130 - NER: 102/152 – all_Bo_GDDT_ma_phuong_thuc_xet_tuyen_2025
2026-04-18 18:07:22.521 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca48740 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:23.691 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e08f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:23.691 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_Bo_GDDT_ma_phuong_thuc_xet_tuyen_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:23.691 | INFO     | src.services.ner_service:extract_batch:130 - NER: 103/152 – csdt_admin_Bo_GDDT_ma_to_hop_xet_tuyen_2025
2026-04-18 18:07:24.892 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b5c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:26.067 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0d61e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:26.067 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_Bo_GDDT_ma_to_hop_xet_tuyen_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:26.068 | INFO     | src.services.ner_service:extract_batch:130 - NER: 104/152 – thi_sinh_Bo_GDDT_ho_so_minh_chung_uu_tien_doi_tuong_2025
2026-04-18 18:07:27.248 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0d61b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:28.409 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d3abf20 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:28.409 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_ho_so_minh_chung_uu_tien_doi_tuong_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:28.409 | INFO     | src.services.ner_service:extract_batch:130 - NER: 105/152 – thi_sinh_Bo_GDDT_khai_bao_uu_tien_khu_vuc_2025
2026-04-18 18:07:29.629 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca98830 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:30.882 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca99190 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:30.882 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_khai_bao_uu_tien_khu_vuc_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:30.882 | INFO     | src.services.ner_service:extract_batch:130 - NER: 106/152 – csdt_admin_Bo_GDDT_xac_nhan_uu_tien_khu_vuc_2025
2026-04-18 18:07:32.087 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b7350 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:33.257 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b9e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:33.257 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_Bo_GDDT_xac_nhan_uu_tien_khu_vuc_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:33.257 | INFO     | src.services.ner_service:extract_batch:130 - NER: 107/152 – thi_sinh_Bo_GDDT_tranh_cong_diem_uu_tien_2_lan_2025
2026-04-18 18:07:34.429 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9ea5d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:35.600 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca56d50 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:35.600 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_tranh_cong_diem_uu_tien_2_lan_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:35.601 | INFO     | src.services.ner_service:extract_batch:130 - NER: 108/152 – all_Bo_GDDT_nguyen_tac_quy_doi_diem_2025
2026-04-18 18:07:36.768 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca0a360 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:37.951 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca09340 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:37.951 | WARNING  | src.services.ner_service:extract:89 - NER failed for all_Bo_GDDT_nguyen_tac_quy_doi_diem_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:37.951 | INFO     | src.services.ner_service:extract_batch:130 - NER: 109/152 – csdt_admin_Bo_GDDT_trach_nhiem_quy_doi_thi_rieng_2025
2026-04-18 18:07:39.117 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca558b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:40.287 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca16a80 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:40.287 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_Bo_GDDT_trach_nhiem_quy_doi_thi_rieng_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:40.287 | INFO     | src.services.ner_service:extract_batch:130 - NER: 110/152 – csdt_admin_Bo_GDDT_lich_xu_ly_nguyen_vong_dot_1_2025
2026-04-18 18:07:41.491 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca65490 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:42.723 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca48c20 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:42.723 | WARNING  | src.services.ner_service:extract:89 - NER failed for csdt_admin_Bo_GDDT_lich_xu_ly_nguyen_vong_dot_1_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:42.724 | INFO     | src.services.ner_service:extract_batch:130 - NER: 111/152 – thi_sinh_Bo_GDDT_dinh_nghia_khu_vuc_1_2025
2026-04-18 18:07:43.929 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca544a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:45.103 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca48ad0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:45.104 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_dinh_nghia_khu_vuc_1_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:45.104 | INFO     | src.services.ner_service:extract_batch:130 - NER: 112/152 – thi_sinh_Bo_GDDT_minh_chung_uu_tien_doi_tuong_01_2025
2026-04-18 18:07:46.269 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7b7a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:47.440 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca8be60 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:47.440 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_minh_chung_uu_tien_doi_tuong_01_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:47.440 | INFO     | src.services.ner_service:extract_batch:130 - NER: 113/152 – thi_sinh_Bo_GDDT_minh_chung_uu_tien_doi_tuong_07_giao_vien_ysy_2025
2026-04-18 18:07:48.614 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca14a10 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:49.783 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e2090 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:49.783 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_minh_chung_uu_tien_doi_tuong_07_giao_vien_ysy_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:49.783 | INFO     | src.services.ner_service:extract_batch:130 - NER: 114/152 – thi_sinh_Bo_GDDT_thay_doi_che_do_uu_tien_khu_vuc_2025
2026-04-18 18:07:51.006 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0cc410 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:52.202 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e8950 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:52.202 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Bo_GDDT_thay_doi_che_do_uu_tien_khu_vuc_2025: All LLM providers failed for JSON mode
2026-04-18 18:07:52.202 | INFO     | src.services.ner_service:extract_batch:130 - NER: 115/152 – chunked_6_chunk_0
2026-04-18 18:07:53.451 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0e9340 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:54.686 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e2ae0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:54.686 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:07:54.686 | INFO     | src.services.ner_service:extract_batch:130 - NER: 116/152 – chunked_6_chunk_1
2026-04-18 18:07:55.895 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0e1f40 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:57.068 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca320c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:57.068 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_1: All LLM providers failed for JSON mode
2026-04-18 18:07:57.069 | INFO     | src.services.ner_service:extract_batch:130 - NER: 117/152 – chunked_6_chunk_2
2026-04-18 18:07:58.255 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca88fb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:59.435 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0cf170 state=finished raised PermissionDeniedError>]
2026-04-18 18:07:59.435 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_2: All LLM providers failed for JSON mode
2026-04-18 18:07:59.435 | INFO     | src.services.ner_service:extract_batch:130 - NER: 118/152 – chunked_6_chunk_3
2026-04-18 18:08:00.606 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca30f50 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:01.775 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca14a10 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:01.775 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_3: All LLM providers failed for JSON mode
2026-04-18 18:08:01.775 | INFO     | src.services.ner_service:extract_batch:130 - NER: 119/152 – chunked_6_chunk_4
2026-04-18 18:08:02.953 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0e84a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:04.112 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca78950 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:04.113 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_4: All LLM providers failed for JSON mode
2026-04-18 18:08:04.113 | INFO     | src.services.ner_service:extract_batch:130 - NER: 120/152 – chunked_6_chunk_5
2026-04-18 18:08:05.326 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0bb6b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:06.566 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca54740 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:06.567 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_5: All LLM providers failed for JSON mode
2026-04-18 18:08:06.567 | INFO     | src.services.ner_service:extract_batch:130 - NER: 121/152 – chunked_6_chunk_6
2026-04-18 18:08:07.770 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca66390 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:08.941 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0b6e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:08.941 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_6: All LLM providers failed for JSON mode
2026-04-18 18:08:08.942 | INFO     | src.services.ner_service:extract_batch:130 - NER: 122/152 – chunked_6_chunk_7
2026-04-18 18:08:10.111 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca54440 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:11.286 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d815040 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:11.286 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_7: All LLM providers failed for JSON mode
2026-04-18 18:08:11.286 | INFO     | src.services.ner_service:extract_batch:130 - NER: 123/152 – chunked_6_chunk_8
2026-04-18 18:08:12.475 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca08bc0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:13.639 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca30260 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:13.640 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_8: All LLM providers failed for JSON mode
2026-04-18 18:08:13.640 | INFO     | src.services.ner_service:extract_batch:130 - NER: 124/152 – chunked_6_chunk_9
2026-04-18 18:08:14.829 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9eac60 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:16.001 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0d4950 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:16.001 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_9: All LLM providers failed for JSON mode
2026-04-18 18:08:16.001 | INFO     | src.services.ner_service:extract_batch:130 - NER: 125/152 – chunked_6_chunk_10
2026-04-18 18:08:17.201 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0d4fe0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:18.442 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca65040 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:18.442 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_10: All LLM providers failed for JSON mode
2026-04-18 18:08:18.442 | INFO     | src.services.ner_service:extract_batch:130 - NER: 126/152 – chunked_6_chunk_11
2026-04-18 18:08:19.643 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0cfa10 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:20.834 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0f7b00 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:20.834 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_11: All LLM providers failed for JSON mode
2026-04-18 18:08:20.834 | INFO     | src.services.ner_service:extract_batch:130 - NER: 127/152 – chunked_6_chunk_12
2026-04-18 18:08:22.056 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0f5910 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:23.273 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b7e30 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:23.273 | WARNING  | src.services.ner_service:extract:89 - NER failed for chunked_6_chunk_12: All LLM providers failed for JSON mode
2026-04-18 18:08:23.273 | INFO     | src.services.ner_service:extract_batch:130 - NER: 128/152 – thi_sinh_Khoa_Hoc_thong_tin_lien_he_chung_2025
2026-04-18 18:08:24.445 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b6de0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:25.611 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e9bb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:25.611 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_thong_tin_lien_he_chung_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:25.611 | INFO     | src.services.ner_service:extract_batch:130 - NER: 129/152 – thi_sinh_Khoa_Hoc_dieu_kien_xet_tuyen_thang_2025
2026-04-18 18:08:26.784 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b4980 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:27.953 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d379280 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:27.953 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_dieu_kien_xet_tuyen_thang_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:27.953 | INFO     | src.services.ner_service:extract_batch:130 - NER: 130/152 – thi_sinh_Khoa_Hoc_cong_thuc_xet_tuyen_thpt_hb_2025
2026-04-18 18:08:29.166 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049d3abd10 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:30.400 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0f7800 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:30.400 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_cong_thuc_xet_tuyen_thpt_hb_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:30.401 | INFO     | src.services.ner_service:extract_batch:130 - NER: 131/152 – thi_sinh_Khoa_Hoc_cong_thuc_xet_tuyen_kien_truc_2025
2026-04-18 18:08:31.608 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c9eb3e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:32.780 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca645c0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:32.780 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_cong_thuc_xet_tuyen_kien_truc_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:32.780 | INFO     | src.services.ner_service:extract_batch:130 - NER: 132/152 – thi_sinh_Khoa_Hoc_quy_dinh_nang_khieu_ve_2025
2026-04-18 18:08:33.954 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca09e20 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:35.117 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca00cb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:35.117 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_quy_dinh_nang_khieu_ve_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:35.117 | INFO     | src.services.ner_service:extract_batch:130 - NER: 133/152 – thi_sinh_Khoa_Hoc_gioi_han_diem_thuong_2025
2026-04-18 18:08:36.305 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca65f70 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:37.468 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0cee10 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:37.469 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_gioi_han_diem_thuong_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:37.469 | INFO     | src.services.ner_service:extract_batch:130 - NER: 134/152 – thi_sinh_Khoa_Hoc_diem_thuong_thanh_tich_2025
2026-04-18 18:08:38.635 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca00230 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:39.813 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca33c20 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:39.813 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_diem_thuong_thanh_tich_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:39.813 | INFO     | src.services.ner_service:extract_batch:130 - NER: 135/152 – thi_sinh_Khoa_Hoc_diem_thuong_ccnn_2025
2026-04-18 18:08:41.039 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca00dd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:42.295 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e1580 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:42.295 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_diem_thuong_ccnn_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:42.295 | INFO     | src.services.ner_service:extract_batch:130 - NER: 136/152 – thi_sinh_Khoa_Hoc_nguong_dau_vao_kien_truc_2025
2026-04-18 18:08:43.523 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca32270 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:44.698 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca787d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:44.698 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_nguong_dau_vao_kien_truc_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:44.699 | INFO     | src.services.ner_service:extract_batch:130 - NER: 137/152 – thi_sinh_Khoa_Hoc_quy_doi_ccnn_thpt_2025
2026-04-18 18:08:45.883 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0e3980 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:47.058 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0e1ac0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:47.058 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_quy_doi_ccnn_thpt_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:47.058 | INFO     | src.services.ner_service:extract_batch:130 - NER: 138/152 – thi_sinh_Khoa_Hoc_gioi_han_chi_tieu_xet_tuyen_thang_2025
2026-04-18 18:08:48.233 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca32bd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:49.403 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c1052e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:49.404 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_gioi_han_chi_tieu_xet_tuyen_thang_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:49.404 | INFO     | src.services.ner_service:extract_batch:130 - NER: 139/152 – thi_sinh_Khoa_Hoc_quy_dinh_kiem_tra_ho_so_2025
2026-04-18 18:08:50.638 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca88f80 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:51.819 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0fdeb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:51.819 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_quy_dinh_kiem_tra_ho_so_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:51.820 | INFO     | src.services.ner_service:extract_batch:130 - NER: 140/152 – thi_sinh_Khoa_Hoc_hoc_phi_du_kien_2025
2026-04-18 18:08:53.051 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0ff500 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:54.323 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca022a0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:54.323 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_hoc_phi_du_kien_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:54.323 | INFO     | src.services.ner_service:extract_batch:130 - NER: 141/152 – thi_sinh_Khoa_Hoc_hoc_bong_diem_dau_vao_cao_2025
2026-04-18 18:08:55.549 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca89460 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:56.718 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca7bda0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:56.718 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_hoc_bong_diem_dau_vao_cao_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:56.718 | INFO     | src.services.ner_service:extract_batch:130 - NER: 142/152 – thi_sinh_Khoa_Hoc_hoc_bong_nganh_khoa_hoc_tu_nhien_2025
2026-04-18 18:08:57.894 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca323f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:59.063 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca4ade0 state=finished raised PermissionDeniedError>]
2026-04-18 18:08:59.063 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_hoc_bong_nganh_khoa_hoc_tu_nhien_2025: All LLM providers failed for JSON mode
2026-04-18 18:08:59.064 | INFO     | src.services.ner_service:extract_batch:130 - NER: 143/152 – thi_sinh_Khoa_Hoc_hoc_bong_nganh_lich_su_dong_phuong_2025
2026-04-18 18:09:00.255 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca7aab0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:01.432 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca331d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:01.432 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_Khoa_Hoc_hoc_bong_nganh_lich_su_dong_phuong_2025: All LLM providers failed for JSON mode
2026-04-18 18:09:01.432 | INFO     | src.services.ner_service:extract_batch:130 - NER: 144/152 – thi_sinh_khoahoc_qd_310_tran_tang_hoc_phi_20_2024
2026-04-18 18:09:02.611 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca4b0b0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:03.776 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca0ac30 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:03.777 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoahoc_qd_310_tran_tang_hoc_phi_20_2024: All LLM providers failed for JSON mode
2026-04-18 18:09:03.777 | INFO     | src.services.ner_service:extract_batch:130 - NER: 145/152 – thi_sinh_khoahoc_qd_310_tie_breaker_gpa12_2024
2026-04-18 18:09:04.964 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca64fb0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:06.217 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca54fe0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:06.217 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoahoc_qd_310_tie_breaker_gpa12_2024: All LLM providers failed for JSON mode
2026-04-18 18:09:06.217 | INFO     | src.services.ner_service:extract_batch:130 - NER: 146/152 – thi_sinh_khoahoc_qd_310_hoc_bong_28_plus_2024
2026-04-18 18:09:07.429 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca09fd0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:08.609 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c9e82f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:08.609 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoahoc_qd_310_hoc_bong_28_plus_2024: All LLM providers failed for JSON mode
2026-04-18 18:09:08.609 | INFO     | src.services.ner_service:extract_batch:130 - NER: 147/152 – thi_sinh_khoahoc_qd_310_hoc_bong_26_28_2024
2026-04-18 18:09:09.786 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca564e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:10.947 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b4560 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:10.947 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoahoc_qd_310_hoc_bong_26_28_2024: All LLM providers failed for JSON mode
2026-04-18 18:09:10.947 | INFO     | src.services.ner_service:extract_batch:130 - NER: 148/152 – thi_sinh_khoahoc_qd_310_hoc_bong_khong_tinh_uu_tien_2024
2026-04-18 18:09:12.130 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049ca0bb90 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:13.297 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049ca57aa0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:13.298 | WARNING  | src.services.ner_service:extract:89 - NER failed for thi_sinh_khoahoc_qd_310_hoc_bong_khong_tinh_uu_tien_2024: All LLM providers failed for JSON mode
2026-04-18 18:09:13.298 | INFO     | src.services.ner_service:extract_batch:130 - NER: 149/152 – 2_17963d0e_0001_chunk_0
2026-04-18 18:09:14.478 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b5820 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:15.657 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c107560 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:15.657 | WARNING  | src.services.ner_service:extract:89 - NER failed for 2_17963d0e_0001_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:09:15.657 | INFO     | src.services.ner_service:extract_batch:130 - NER: 150/152 – 2_17963d0e_0002_chunk_0
2026-04-18 18:09:16.831 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c0b5850 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:18.034 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049d5351f0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:18.035 | WARNING  | src.services.ner_service:extract:89 - NER failed for 2_17963d0e_0002_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:09:18.035 | INFO     | src.services.ner_service:extract_batch:130 - NER: 151/152 – 2_17963d0e_0003_chunk_0
2026-04-18 18:09:19.296 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c11c3e0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:20.464 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c105550 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:20.464 | WARNING  | src.services.ner_service:extract:89 - NER failed for 2_17963d0e_0003_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:09:20.464 | INFO     | src.services.ner_service:extract_batch:130 - NER: 152/152 – 2_17963d0e_0004_chunk_0
2026-04-18 18:09:21.664 | WARNING  | src.services.llm_client:chat_json:244 - LLM JSON [ramclouds] json_mode=True failed: RetryError[<Future at 0x7c049c11cbf0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:22.829 | WARNING  | src.services.llm_client:chat_json:253 - LLM JSON [ramclouds] json_mode=False fallback failed: RetryError[<Future at 0x7c049c0b74d0 state=finished raised PermissionDeniedError>]
2026-04-18 18:09:22.829 | WARNING  | src.services.ner_service:extract:89 - NER failed for 2_17963d0e_0004_chunk_0: All LLM providers failed for JSON mode
2026-04-18 18:09:22.829 | INFO     | __main__:main:178 - NER: 0 success, 152 failed
Traceback (most recent call last):
  File "/content/husc-admission-chat-enrollment/rag2025/scripts/build_graph.py", line 241, in <module>
    asyncio.run(main(dry_run=args.dry_run, limit=args.limit, incremental=args.incremental))
  File "/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/content/husc-admission-chat-enrollment/rag2025/scripts/build_graph.py", line 181, in main
    raise RuntimeError(
RuntimeError: NER extraction failed for all chunks. Check provider credentials/model access and rerun graph build.

❌ STEP_FAILED
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_397/1386607063.py in <cell line: 0>()
     43 run_step('Step 1: LanceDB Ingest', ['python', 'scripts/ingest_lancedb.py'])
     44 os.environ['HF_HOME'] = '/content/hf_cache'
---> 45 run_step('Step 2: Build Knowledge Graph', ['python', 'scripts/build_graph.py'])
     46 run_step('Step 3: Verification', ['python', 'scripts/preflight_check.py'])

/tmp/ipykernel_397/1386607063.py in run_step(step_name, cmd)
     38     if proc.returncode != 0:
     39         print('❌ STEP_FAILED')
---> 40         raise RuntimeError(f'{step_name} failed')
     41 
     42 # Execute steps

RuntimeError: Step 2: Build Knowledge Graph failed


Thông báo lỗi 


# --- BÁO CÁO LỖI HIỆN TẠI (DIAGNOSTIC REPORT) ---
# Lỗi xảy ra tại: Step 2: Build Knowledge Graph
# Script: scripts/build_graph.py
# 
# KHOANH VÙNG CÁC CHỖ CẦN KIỂM TRA:
# 1. Lỗi chính: RuntimeError: NER extraction failed for all chunks.
# 2. Nguyên nhân gốc rễ (Traceback): PermissionDeniedError tại src.services.llm_client.
# 3. Trạng thái NER: 0 success, 152 failed.
# 
# GỢI Ý FIX:
# - Kiểm tra lại RAMCLOUDS_API_KEY trong Colab Secrets xem có còn hạn mức không.
# - Model 'gpt-5.4' có thể đang từ chối request (403 Forbidden/Permission Denied).
# - Thử đổi model khác trong cell cấu hình nếu gpt-5.4 không khả dụng.


Bạn giúp tôi test trên cái key này: sk-d3nCvg0HfZ10Kw5i4hSiWFcvPTKrane52OJXPlQsiUExyFEe
Xem thử là gpt-5.4 ở đây có khả dụng hay không
