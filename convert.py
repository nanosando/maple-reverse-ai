
import onnx
import torch

from reverse.ReverseGame import ReverseGame
from reverse.ReversePlayers import *
from reverse.NNet import NNetWrapper as NNet


cuda = torch.cuda.is_available()
g = ReverseGame(8)
nnet = NNet(g)
nnet.load_checkpoint('./pretrained_models/', 'best.pth')

board = g.getInitBoard()
can_board = g.getCanonicalForm(board, 1)
model_input = torch.FloatTensor(can_board.astype(np.float64))
if cuda: model_input = model_input.contiguous().cuda()
model_input = model_input.view(1, 8, 8)

model = nnet.nnet
model.eval()

# torch2onnx
onnx_model_path = './pretrained_models/best.onnx'
torch.onnx.export(
    model,
    model_input, 
    onnx_model_path,
    input_names=['input'],
    output_names=['pi', 'v']
)

model_onnx = onnx.load(onnx_model_path)
onnx.checker.check_model(model_onnx)
