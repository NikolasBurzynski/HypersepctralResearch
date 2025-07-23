import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch





class ManualCombination:
    def __init__(self, img, designMatrix, net, testDataBundle, device, MODEL_PATH, weighted = False, weights = None, wave_numbers = 17):
        testData = testDataBundle[0][:, 2:wave_numbers+2]
        # testDataName = testDataBundle[1]
        IM_SIZE = testDataBundle[2]
        self.point = (0,0)
        self.counter = 0
        img = (img/np.max(img) * 255).astype(np.uint8)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        self.original = self.img.copy()
        self.designMatrix = designMatrix.cpu().numpy()
        self.net = net
        self.testData = testData
        self.device = device
        self.MODEL_PATH = MODEL_PATH
        self.IM_HEIGHT = IM_SIZE[0]
        self.IM_WIDTH = IM_SIZE[1]
        self.weights = weights
        self.weighted = weighted
        self.wave_numbers = wave_numbers
        self.bce_loss = torch.nn.BCELoss()

        # self.manual_dna = 0
        self.manual_bsa = 0
        self.manual_oa = 0
        self.manual_bg = 0

        self.create_graph()

    def calc_unmixing_loss(self, goal, res_spec, weights): # Weighted square error
        # TODO: Implement this without tensor flow
        # weights = torch.tensor(weights).to(self.device)
        # inputs = torch.tensor(inputs).to(self.device)
        # design_matrix = torch.tensor(self.designMatrix).to(self.device)
        # resSpec = torch.mm(outputs, design_matrix)
        # res = torch.sum(((inputs-resSpec) * weights) ** 2, axis = 1)
        # return torch.sum(res)
        square_diff = np.sum(((goal - res_spec) * weights) ** 2)
        return square_diff
    
    def mousePoints(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x,y)
            self.counter = self.counter + 1

            self.create_graph()
        
        if event == cv2.EVENT_RBUTTONDOWN:
            x = int(input("Enter x cord: "))
            y = int(input("Enter y cord: "))
            self.point = (x,y)
            self.counter = self.counter + 1

            self.create_graph()

    def create_graph(self):
        figure = plt.figure(2)
        plt.clf()
        sOA = self.designMatrix[0] * self.manual_oa
        sBSA = self.designMatrix[1] * self.manual_bsa
        sBG = self.designMatrix[2] * self.manual_bg
        # resultSpec = sDNA + sBSA + sOA + sBG
        resultSpec = sBSA + sOA + sBG

        pixel = self.testData[self.IM_WIDTH*self.point[1] + self.point[0], :]
        new_pixel = pixel - np.min(pixel)
        unmixing_loss = self.calc_unmixing_loss(pixel, resultSpec, self.weights)
        
        print(f"weighted_se_LOSS: {unmixing_loss}\t BSA:{self.manual_bsa}\t OA:{self.manual_oa}\t BG:{self.manual_bg}", end='\r')
        
        plt.plot(sBSA, color = "blue")
        plt.plot(sOA, color = "green")
        plt.plot(sBG, color = "black")
        plt.plot(resultSpec)
        plt.plot(pixel)
        plt.plot(new_pixel)

        self.manual_graph = self.convert_plt2cv2(figure)

    def onBSA(self, val):
        self.manual_bsa = .001*val
        self.create_graph()
    
    def onOA(self, val):
        self.manual_oa = .001*val
        self.create_graph()

    def onBG(self, val):
        self.manual_bg = .001*val
        self.create_graph()

    @staticmethod
    def convert_plt2cv2(fig):
        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        return img

    def test(self):
        checkpoint = torch.load(self.MODEL_PATH)
        self.net.load_state_dict(checkpoint["Model_State_Dict"])
        
        x, y = self.point

        pixel = np.array([self.testData[self.IM_WIDTH*y + x, :]])
        self.net.eval()

        dna_output, unmixing_output = self.net(torch.from_numpy(pixel).type(torch.FloatTensor).to(device = self.device))

        dna_result = dna_output.detach().cpu().numpy()[0]
        unmixing_result = unmixing_output.detach().cpu().numpy()[0]
        goal = pixel[0]

        sOA = self.designMatrix[0] * unmixing_result[0]
        sBSA = self.designMatrix[1] * unmixing_result[1]
        sBG = self.designMatrix[2] * unmixing_result[2]
        # resultSpec = sDNA + sBSA + sOA + sBG
        resultSpec = sBSA + sOA + sBG
        
        unmixing_loss = self.calc_unmixing_loss(goal, resultSpec, self.weights)
        print("")
        print(f"DNA_Prob: {dna_result} weighted_se_LOSS: {unmixing_loss} OA: {unmixing_result[0]}  BSA: {unmixing_result[1]} BG: {unmixing_result[2]}", end='\n')

        return (sOA, sBSA, sBG), resultSpec, goal


    def eventLoop(self):
        fig = plt.figure(1)
        nn_graph = np.zeros((256,256))
        manual_graph = np.zeros((256,256))

        cv2.namedWindow("Manual Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Manual Result", 750, 750)


        cv2.imshow("Manual Result", manual_graph)
        cv2.createTrackbar('BSA_slider', "Manual Result", 0, 1000, self.onBSA)
        cv2.createTrackbar('OA_slider', "Manual Result", 0, 1000, self.onOA)
        cv2.createTrackbar('BG_slider', "Manual Result", 0, 1000, self.onBG)
        while True:

            if self.counter == 1:
                x, y = self.point

                # Draw rectangle for area of interest

                self.img = self.original.copy()

                cv2.rectangle(self.img, (x-2, y-2), (x+2, y+2), (255, 255, 255), 1)
                cv2.imshow("Original Image ", self.img)
                # cv2.imsave("Image.png", self.img)

                sEMS, resultSpec, goalSpec = self.test()
                plt.figure(1)
                plt.clf()
                plt.plot(sEMS[0], label = "OA", color = "green")
                plt.plot(sEMS[1], label = "BSA", color = "blue")
                plt.plot(sEMS[2], label = "BG", color = "black")
                plt.plot(resultSpec, label = "Result")
                plt.plot(goalSpec, label = "Goal")
                plt.legend()

                nn_graph = self.convert_plt2cv2(fig)

                self.counter = 0

            cv2.imshow("Original Image ", self.img)
            cv2.imshow("Neural Net Result", nn_graph)
            cv2.imshow("Manual Result", self.manual_graph)


            cv2.setMouseCallback("Original Image ", self.mousePoints)


            key = cv2.waitKey(1)    
            if key == ord('q'):
                break