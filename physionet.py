import os
import tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
import utils


## returns max, min val feature list among records
def get_data_min_max(records, device):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    ## loop for each record_id == patient
    for b, (record_id, tt, vals, mask, lbaels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        ## extract max min val of each feature
        for i in range(n_features):
            existing_vals = vals[:, i][mask[:, i] == 1]
            if len(existing_vals) != 0:
                batch_max.append(torch.max(existing_vals))
                batch_min.append(torch.min(existing_vals))
            else:
                batch_max.append(-inf)
                batch_min.append(inf)
        batch_max = torch.stack(batch_max)
        batch_min = torch.stack(batch_min)

        if (data_min is None) and (data_max is None):
            data_max = batch_max
            data_min = batch_min
        else:
            data_max = torch.max(data_max, batch_max)
            data_min = torch.min(data_min, batch_min)

    return data_min, data_max


class PhysioNet(object):
    ## time series data
    urls = [
        "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
        "https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download",
    ]

    ## outcomes for each pateint
    outcome_urls = ["https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt"]

    params = [
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "Weight",
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
    ]

    # key : params, value: index of parmams
    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]

    # key : labels, value: index of labels
    labels_dict = {k: i for i, k in enumerate(labels)}

    # initializing
    def __init__(
        self,
        root,
        train=True,
        download=False,
        quantization=0.1,
        n_samples=None,
        device=torch.device("cpu"),
    ):
        self.root = root  # data path
        self.train = train  # training set use or not
        self.device = device  # device to load data
        self.reduce = "average"  #같은 시간대의 같은 특성은 평균 값 사용
        self.quantization = quantization  # time units

        # download if there is no data
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("No data set. Set download = True")

        # set appropriate file as data_file
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # load data and labels on appropriate device
        if self.device == "cpu":
            self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location="cpu")
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location="cpu")
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        # truncate samples up to n_samples
        if n_samples is not None:
            self.data = self.data[:n_samples]
            # self.labels = self.labels[:n_samples]

    def download(self):
        if self._check_exists():
            return

        # if folder exists pass else make dir
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            # 뒤에서부터 찾아서 / 기준으로 3개의 구역으로 분리
            filename = url.rpartition("/")[2]
            # Download a file from a url and place it in root.
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                # 데이터 시작 지점부터 읽음
                for l in lines[1:]:
                    l = l.rstrip().split(",")
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    outcomes,  ##labels
                    os.path.join(self.processed_folder, filename.split(".")[0] + ".pt"),
                )

        for url in self.urls:
            filename = url.rpartition("/")[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            # raw 폴더에 압축 해제
            tar.extractall(self.raw_folder)
            tar.close()

            print(f"Processing {filename}")

            dirname = os.path.join(self.raw_folder, filename.split(".")[0])  # 그냥 이름이 set-a이렇게 생김
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split(".")[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.0]  # time trace
                    vals = [torch.zeros(len(self.params)).to(self.device)]  # 관측값
                    mask = [torch.zeros(len(self.params)).to(self.device)]  # 존재여부
                    nobs = [torch.zeros(len(self.params))]  # 관측 횟수
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(",")
                        # hours
                        time = float(time.split(":")[0]) + float(time.split(":")[1]) / 60.0
                        # 계산 속도 향상을 위한 시간 단위 부여
                        time = round(time / self.quantization) * self.quantization

                        # 매 시간 마다 각 변수에 새로운 값 할당
                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)).to(self.device))
                            mask.append(torch.zeros(len(self.params)).to(self.device))
                            nobs.append(torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            n_observations = nobs[-1][self.params_dict[param]]
                            # 같은 시간에 한개 이상의 관측값이 있는 경우, 평균값을 사용
                            if self.reduce == "average" and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)

                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == "RecordID", f"Unexpected param {param}"

                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # for training set take mortality data
                    labels = outcomes[record_id]
                    labels = labels[4]

                # 각 환자 데이터에 대한 정보 저장
                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(
                    self.processed_folder,
                    filename.split(".")[0] + "_" + str(self.quantization) + ".pt",
                ),
            )

        print("Done !")

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition("/")[2]

            if not os.path.exists(
                os.path.join(
                    self.processed_folder,
                    filename.split(".")[0] + "_" + str(self.quantization) + ".pt")
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def training_file(self):
        return "set-a_{}.pt".format(self.quantization)

    @property
    def test_file(self):
        return "set-b_{}.pt".format(self.quantization)

    @property
    def label_file(self):
        return "Outcomes-a.pt"
    # 대괄호를 사용한 인덱싱, 슬라이싱 정의
    def __getitem__(self, index):
        return self.data[index]
    # 객체 길이 정의
    def __len__(self):
        return len(self.data)





if __name__ == '__main__':
    torch.manual_seed(1991) 

    dataset = PhysioNet('data/physionet', train=False, download=True)
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
    # print(dataloader.__iter__().next())
    # print(len(dataset))