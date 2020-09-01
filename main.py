from paddleocr import PaddleOCR

from gui.tray import TrayWindowMain

if __name__ == '__main__':
    paddle = PaddleOCR(det_model_dir='./interface/ch_det_mv3_db/', rec_model_dir='./interface/ch_rec_mv3_crnn/', use_gpu=False)

    TrayWindowMain(paddle)
