# python train.py --config-file configs/diverse_weather.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather/model_best.pth

CUDA_VISIBLE_DEVICES=1 python train.py --config-file configs/diverse_weather_foggy_test.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather_batch4_eval1000/model_best.pth > diverse_weather_foggy_test.log
CUDA_VISIBLE_DEVICES=1 python train.py --config-file configs/diverse_weather_dusk_rainy_test.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather_batch4_eval1000/model_best.pth > diverse_weather_dusk_rainy_test.log
CUDA_VISIBLE_DEVICES=1 python train.py --config-file configs/diverse_weather_night_rainy_test.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather_batch4_eval1000/model_best.pth > diverse_weather_night_rainy_test.log
CUDA_VISIBLE_DEVICES=1 python train.py --config-file configs/diverse_weather_night_sunny_test.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather_batch4_eval1000/model_best.pth > diverse_weather_night_sunny_test.log


