## NEW RESULTS

# On OxfordPet, 2 classes
# {'acc': array(0.92255694, dtype=float32), 'fscore': array(0.8717927, dtype=float32), 'jaccard': array(0.8042548, dtype=float32), 'miou': array(0.84533525, dtype=float32), 'acc_class': array(0.9244877, dtype=float32)}

# On OxfordPed, 3 classes
# acc: 0.809112012386322, fscore: 0.6950455904006958, jaccard: 0.4236198961734772, miou: 0.5895310044288635, acc_class: 0.7907080054283142,

#ON MSRA
# acc: 0.8870965242385864, fscore: 0.7340942621231079, jaccard: 0.5693441033363342, miou: 0.7183158993721008, acc_class: 0.8222122192382812,

#ON ECSSD: acc: 0.8660160303115845, fscore: 0.7175257205963135, jaccard: 0.5610008239746094, miou: 0.6996684074401855, acc_class: 0.8161179423332214,

## OLD RESULTS

# OxfordPet 3 classes, new results
# CE: fscore: 0.9406614303588867, jaccard: 0.803737998008728, miou: 0.8626742362976074, acc_class: 0.9209012389183044,
# with normalization: acc: 0.8007621169090271, fscore: 0.7177091836929321, jaccard: 0.4483543634414673, miou: 0.5951331257820129, acc_class: 0.7891793251037598,
# without normalization: fscore: 0.09686852246522903, jaccard: 0.004766244441270828, miou: 0.4686095714569092, acc_class: 0.6112871170043945,
# gauss on 3 classes fscore: 0.6184778213500977, jaccard: 0.3469609320163727, miou: 0.5122430324554443, acc_class: 0.7313703298568726,
# gauss on 3 classes with class prob estimated from data fscore: 0.614776611328125, jaccard: 0.3450876772403717, miou: 0.51039057970047, acc_class: 0.7302936315536499,

# MobileV2, CE 94F, Yuri's model 78F, 2 classes, without normalization, 0.83, multiply by 10: 0.81075454
# old result: MobileV2, CE 0.847, Yuri's model miou: 0.7687325477600098, without miou: 0.8063000440597534
# With normalization: 0.80108523, without final normalizatin 0.8275003
# With Gaussian: fscore: 0.8597172498703003, jaccard: 0.7880023717880249, miou: 0.8317950367927551, acc_class: 0.9173038601875305,



# ON MSRA: CE .847, miou: .814
# Yuri's model  with norm'fscore': fscore: 0.5401815176010132, jaccard: 0.4674459993839264, miou: 0.5906269550323486, acc_class: 0.838834285736084,
# without normalization fscore': array(0.67282724, dtype=float32), 'jaccard': array(0.54889095, dtype=float32), 'miou': array(0.6930462, dtype=float32), 'acc_class': array(0.83573437, dtype=float32)}
# Gauss: fscore: 0.7196846008300781, jaccard: 0.5063784718513489, miou: 0.6829910278320312, acc_class: 0.7751867771148682,

# ON ECSD
# CE: fscore: 0.8170745372772217, jaccard: 0.6717592477798462, miou: 0.7803574204444885,
# fscore: 0.6602092981338501, jaccard: 0.5078501105308533, miou: 0.6573360562324524, acc_class: 0.791091799736023,
# normalized fscore: 0.5660905838012695, jaccard: 0.4877687692642212, miou: 0.5953315496444702, acc_class: 0.8236201405525208,
# With Gaussian fscore: 0.6886737942695618, jaccard: 0.4657036066055298, miou: 0.6456500291824341, acc_class: 0.7477684020996094,

#ON DUTO
# acc: 0.9190266728401184, fscore: 0.7480570077896118, jaccard: 0.5376801490783691, miou: 0.7241442203521729, acc_class: 0.7976900339126587,
# normalized:  fscore: 0.34994977712631226, jaccard: 0.28593116998672485, miou: 0.4480414092540741, acc_class: 0.7580647468566895,
# Not normalized  fscore: 0.4634014070034027, jaccard: 0.28246861696243286, miou: 0.5588617920875549, acc_class: 0.6636601090431213,
# Gauss:  fscore: 0.5237104296684265, jaccard: 0.35254400968551636, miou: 0.5985630750656128, acc_class: 0.7170082330703735,

# ON DUTS
# CE: fscore: 0.7153045535087585, jaccard: 0.550285816192627, miou: 0.7278420329093933, acc_class: 0.8263756036758423,
# normalized fscore': array(0.4005228, dtype=float32), 'jaccard': array(0.3345051, dtype=float32), 'miou': array(0.5060257, dtype=float32), 'acc_class': array(0.8128376, dtype=float32)}
# not normalized fscore: 0.5133204460144043, jaccard: 0.41974586248397827, miou: 0.6137796640396118, acc_class: 0.828128457069397,
#fscore: 0.5997424721717834, jaccard: 0.4523462653160095, miou: 0.65932697057724, acc_class: 0.7934781312942505,

