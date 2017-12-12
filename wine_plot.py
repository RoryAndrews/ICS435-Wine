import matplotlib.pyplot as plt


def plot_params(params):
    pass

params = dict()

params['red-unscaled']['C'] = [
    5.3366992312063068,
    5.3366992312063068,
    8.5203287155197067,
    16.0,
    1.0
]
params['red-unscaled']['epsilon'] = [
    0.36363636363636365,
    0.18181818181818182,
    0.090909090909090912,
    0.77777777777777768,
    0.1111111111111111
]
params['red-unscaled']['gamma'] = [
    0.01,
    0.01,
    0.0009765625,
    0.0009765625,
    0.0009765625
]



params['red-max-abs-scaled']['C'] = [
    15.199110829529332,
    0.65793322465756787,
    0.091217506605090412,
    0.0625,
    0.015625
]
params['red-max-abs-scaled']['epsilon'] = [
    0.18181818181818182,
    0.0,
    0.63636363636363635,
    0.66666666666666663,
    0.88888888888888884
]
params['red-max-abs-scaled']['gamma'] = [
    0.23101297000831592,
    5.3366992312063068,
    2.7407019694402481,
    4.0,
    1.0
]



params['white-unscaled']['C'] = [
    0.65793322465756787,
    0.01,
    0.0009765625,
    0.25,
    0.015625
]
params['white-unscaled']['epsilon'] = [
    0.54545454545454541,
    2.0,
    1.0,
    0.88888888888888884,
    0.77777777777777768
]
params['white-unscaled']['gamma'] = [
    0.01,
    0.01,
    0.2835781305488656,
    0.0009765625,
    0.0009765625
]



params['white-max-abs-scaled']['C'] = [
    0.65793322465756787,
    0.65793322465756787,
    0.009438198784409749,
    0.015625,
    0.0009765625
]
params['white-max-abs-scaled']['epsilon'] = [
    0.72727272727272729,
    0.90909090909090917,
    0.72727272727272729,
    0.77777777777777768,
    0.55555555555555558
]
params['white-max-abs-scaled']['gamma'] = [
    5.3366992312063068,
    5.3366992312063068,
    8.5203287155197067,
    4.0,
    4.0
]



params['combined-unscaled']['C'] = [
    1.873817422860383,
    1.873817422860383,
    2.7407019694402481,
    1.0,
    0.0625
]
params['combined-unscaled']['epsilon'] = [
    0.54545454545454541,
    0.0,
    0.0,
    0.22222222222222221,
    0.22222222222222221
]
params['combined-unscaled']['gamma'] = [
    0.01,
    0.23101297000831592,
    0.2835781305488656,
    0.25,
    0.25
]



params['combined-max-abs-scaled']['C'] = [
    1.873817422860383,
    1.873817422860383,
    8.5203287155197067,
    4.0,
    16.0
]
params['combined-max-abs-scaled']['epsilon'] = [
    0.18181818181818182,
    0.0,
    0.0,
    0.1111111111111111,
    0.22222222222222221
]
params['combined-max-abs-scaled']['gamma'] = [
    123.28467394420659,
    123.28467394420659,
    256.0,
    256.0,
    256.0
]


params['red-unscaled']['mae_mean'] = [
    0.539661053232,
    0.518980772145,
    0.531009712691,
    0.565586347723,
    0.534932632591
]
params['red-unscaled']['mae_std'] = [
    0.0173853048779,
    0.0176546821212,
    0.0175429185527,
    0.0199327569877,
    0.015607172491
]


params['red-max-abs-scaled']['mae_mean'] = [
    0.492020349474,
    0.475733370215,
    0.518770529015,
    0.5241190645,
    0.527290032749
]
params['red-max-abs-scaled']['mae_std'] = [
    0.0155148801258,
    0.0158115809843,
    0.0166229677438,
    0.0179645816034,
    0.0161966263948
]


params['white-unscaled']['mae_mean'] = [
    0.621708935647,
    0.628863111326,
    0.631503264564,
    0.630235229443,
    0.635687389556
]
params['white-unscaled']['mae_std'] = [
    0.0112181925472,
    0.0101984433497,
    0.0142944227332,
    0.00929630248201,
    0.0108668461556
]


params['white-max-abs-scaled']['mae_mean'] = [
    0.56154424173,
    0.560471048783,
    0.573809159304,
    0.576866162235,
    0.583044790123
]
params['white-max-abs-scaled']['mae_std'] = [
    0.00775878748234,
    0.00981941829877,
    0.00707492096508,
    0.00863432499737,
    0.00818190963525
]


params['combined-unscaled']['mae_mean'] = [
    0.59913036939,
    0.505704048894,
    0.504151076741,
    0.556734802022,
    0.560690871747
]
params['combined-unscaled']['mae_std'] = [
    0.0095454175432,
    0.0114160625159,
    0.00879357458697,
    0.0100990753456,
    0.00925609722445
]


params['combined-max-abs-scaled']['mae_mean'] = [
    0.490652843524,
    0.465014414861,
    0.454749552541,
    0.481558971566,
    0.509137912998
]
params['combined-max-abs-scaled']['mae_std'] = [
    0.00671061229663,
    0.00794578740325,
    0.00960003752818,
    0.00938678739314,
    0.00936818797037
]


testtype = list(5)
testtype[0] = "default r2_score scoring"
testtype[1] = "mean absolute error scoring"
testtype[2] = "mean absolute error scoring with sample weights"
testtype[2] = "mean squared error scoring with sample weights"
testtype[2] = "mean squared error scoring with heavier sample weights"

for key in params:
    for _i in range(5):
        params[key]['C']
        params[key]['epsilon']
        params[key]['gamma']
