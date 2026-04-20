# Ubuntu 24.04 + IsaacLab + P73 (Reproducible Setup)

이 문서는 `/home/piene/p73`에 구성된 **P73 학습 환경**을 다른 컴퓨터에서 최대한 동일하게 재현하기 위한 가이드입니다.

## 핵심 요약 (현재 머신에서 확인된 사실)

- **워크스페이스 루트**: `/home/piene/p73`
- **폴더 구성(3개)**:
  - `/home/piene/p73/IsaacLab` (IsaacLab source checkout)
  - `/home/piene/p73/isaaclab_p73` (외부 task/algorithm 패키지)
  - `/home/piene/p73/isaacsim` (Isaac Sim pre-built binaries, zip 설치로 보임)
- **IsaacLab 버전**: `2.3.2`  
  - 근거: `/home/piene/p73/IsaacLab/VERSION` 첫 줄이 `2.3.2`
- **Isaac Sim (binary) 버전**: `5.1.0-rc.19+release...`  
  - 근거: `/home/piene/p73/isaacsim/VERSION` 첫 줄이 `5.1.0-rc.19+release.26219...`
- **IsaacLab ↔ Isaac Sim 연결 방식**: `_isaac_sim` **symlink(심볼릭 링크)**가 binary Isaac Sim을 가리킴  
  - 근거: `/home/piene/p73/IsaacLab/_isaac_sim -> /home/piene/p73/isaacsim` (env_lock에 저장)
- **실제 학습 실행 워크플로우**: `conda env p73` 활성화 후, `./isaaclab.sh -p ... train.py` 실행  
  - 예시(사용자 커맨드):
    - `source ~/.bashrc && conda activate p73 && cd /home/piene/p73/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...`

## 매우 중요: “Isaac Sim binary”와 “pip Isaac Sim”이 동시에 존재함

현재 `conda env p73` 안에서 `isaacsim`은 **pip 패키지(isaacsim==5.0.0.0)** 로도 설치되어 있습니다.

- 근거(환경 lock):
  - `pip freeze`에 `isaacsim==5.0.0.0`, `isaacsim-rl==5.0.0.0` 등이 존재
  - `OMNI_KIT_ACCEPT_EULA=YES python -c 'import isaacsim; ...'` 결과에서:
    - `isaacsim.__file__= /home/piene/miniconda3/envs/p73/lib/python3.11/site-packages/isaacsim/__init__.py`

즉, “완전 동일”을 목표로 한다면 **(A) binary Isaac Sim 폴더**, **(B) conda p73 환경의 pip Isaac Sim 패키지**를 **둘 다** 재현해야 합니다.

## Strict lock(엄밀 잠금) 아티팩트 위치

아래 파일들은 현재 머신에서 추출한 “재현용 스냅샷”입니다.

- **Conda explicit spec(엄밀 잠금)**: `/home/piene/p73/isaaclab_p73/env_lock/conda-p73-linux-64.explicit.txt`  
  - `conda create --name p73 --file <file>` 형태로 복원 가능
- **pip freeze 스냅샷**: `/home/piene/p73/isaaclab_p73/env_lock/pip-freeze-p73.txt`
- **pip show 스냅샷(설치 위치 확인용)**:
  - `/home/piene/p73/isaaclab_p73/env_lock/pip-show-isaaclab.txt`
  - `/home/piene/p73/isaaclab_p73/env_lock/pip-show-isaaclab_p73.txt`
  - `/home/piene/p73/isaaclab_p73/env_lock/pip-show-isaacsim.txt`
- **GPU/Driver 스냅샷**: `/home/piene/p73/isaaclab_p73/env_lock/nvidia-smi.txt`  
  - 예: `Driver Version: 570.211.01`, `CUDA Version: 12.8`, `RTX 5090`
- **_isaac_sim 링크 정보**:
  - `/home/piene/p73/isaaclab_p73/env_lock/isaaclab-_isaac_sim.readlink.txt`
  - `/home/piene/p73/isaaclab_p73/env_lock/isaaclab-_isaac_sim.stat.txt`
- **Git 커밋/변경사항**:
  - `/home/piene/p73/isaaclab_p73/env_lock/git-IsaacLab.txt`
  - `/home/piene/p73/isaaclab_p73/env_lock/git-isaaclab_p73.txt`
  - `/home/piene/p73/isaaclab_p73/env_lock/git-diff-IsaacLab.patch`
  - `/home/piene/p73/isaaclab_p73/env_lock/git-diff-isaaclab_p73.patch`

## 재현 방법 (권장 순서)

### 옵션 1) 가장 동일하게: `/home/piene/p73` 통째로 복사

**가장 “완전 동일”에 가깝게** 만들려면, 다른 PC에도 경로를 동일하게 맞추고(즉 `/home/piene/p73`) 폴더를 통째로 복사하는 방식이 유리합니다.

- 복사 대상:
  - `/home/piene/p73/IsaacLab`
  - `/home/piene/p73/isaaclab_p73`
  - `/home/piene/p73/isaacsim`
  - (선택) `/home/piene/p73/IsaacLab/logs`, `/outputs`, `/wandb` 는 용량이 크면 제외
- 이후 “Conda p73 환경”만 별도로 아래 “옵션 2”로 복원하면 됩니다.

### 옵션 2) 재설치/재구축: 커밋 + patch + conda explicit + pip freeze 기반

> 전제: 이 문서와 함께 `env_lock/` 디렉토리(Strict lock 파일들)를 **같이** 들고 있어야 합니다.  
> **중요**: `env_lock/`은 원본 머신에서 추출한 산출물이라, GitHub에서 `isaaclab_p73`를 새로 clone하면 자동으로 따라오지 않을 수 있습니다.  
> 이 경우 원본 머신에서 `env_lock/` 폴더를 별도로 복사해서 새 머신의 `/home/piene/p73/isaaclab_p73/env_lock/`에 넣고 진행하세요.

#### 2.1 디렉토리 생성

- `/home/piene/p73`를 만들고 그 안에 아래 3개 폴더를 둡니다:
  - `IsaacLab/`
  - `isaaclab_p73/`
  - `isaacsim/`

#### 2.2 Isaac Sim (binary, pre-built) 설치 및 symlink

IsaacLab 공식 문서에는 binary 설치 흐름이 아래처럼 명시돼 있습니다(직접 인용):

> “Set up a symbolic link between the installed Isaac Sim root folder and `_isaac_sim` in the Isaac Lab directory.”  
> (`/home/piene/p73/IsaacLab/docs/source/setup/installation/include/src_symlink_isaacsim.rst`)

그리고 Linux 예시로:

> `ln -s ${ISAACSIM_PATH} _isaac_sim`  
> (`.../include/src_symlink_isaacsim.rst`)

현재 머신은 다음과 같이 연결되어 있습니다:

- `/home/piene/p73/IsaacLab/_isaac_sim -> /home/piene/p73/isaacsim`

따라서 다른 PC도 동일하게 맞춥니다.

권장 절차(커맨드 예시, copy-paste):

```bash
# Isaac Sim binary가 /home/piene/p73/isaacsim 에 존재한다고 가정
# (zip로 푼 폴더를 이 경로로 맞추는 것이 '완전 동일'에 유리)

cd /home/piene/p73/IsaacLab

# 기존 _isaac_sim 이 있다면 삭제 후 다시 생성 (강제 갱신)
rm -f /home/piene/p73/IsaacLab/_isaac_sim
ln -s /home/piene/p73/isaacsim /home/piene/p73/IsaacLab/_isaac_sim
```

#### 2.3 IsaacLab / isaaclab_p73: Git commit(리비전) 맞추기 + patch 적용

현재 머신은 아래 commit을 사용하고 있습니다(파일에 기록됨):

- IsaacLab commit: `73930be06597075d6e587af1f0a55e0439931f83`
  - 근거: `env_lock/git-IsaacLab.txt`
- isaaclab_p73 commit: `06c1c58098098c3a803d8e9f98e6d0000388712b`
  - 근거: `env_lock/git-isaaclab_p73.txt`

또한 IsaacLab에는 로컬 수정이 존재합니다(파일에 기록됨):

- 근거: `env_lock/git-diff-IsaacLab.patch`
  - 예시로, `apps/isaaclab.python.kit`에서 URDF importer 버전 “hard pin(강제 고정)”을 제거하는 변경이 포함되어 있고,
  - `scripts/reinforcement_learning/rsl_rl/{train,play}.py`에 `isaaclab_p73` task에 대한 conditional import 및 custom runner 적용이 포함되어 있습니다.

다른 PC에서는 **동일한 patch를 적용**해야 완전히 동일한 동작이 나옵니다.

권장 절차(요약):

- IsaacLab:
  - `git clone` 후, commit `73930be0...`로 checkout
  - `env_lock/git-diff-IsaacLab.patch`를 **unified diff(통합 diff)**로 `git apply` 해서 로컬 변경을 동일하게 맞춤
- isaaclab_p73:
  - `git clone` 후, commit `06c1c580...`로 checkout
  - (필요 시) `env_lock/git-diff-isaaclab_p73.patch`도 동일하게 적용

권장 절차(커맨드 예시, copy-paste):

```bash
# workspace
mkdir -p /home/piene/p73
cd /home/piene/p73

# 0) isaaclab_p73: clone + checkout exact commit
# (env_lock/ 폴더는 원본 머신에서 복사해와서 /home/piene/p73/isaaclab_p73/env_lock/ 에 위치시킨다고 가정)
git clone https://github.com/P73-project/isaaclab_p73.git
cd /home/piene/p73/isaaclab_p73
git checkout 06c1c58098098c3a803d8e9f98e6d0000388712b

# (optional) apply local patch if non-empty
git apply /home/piene/p73/isaaclab_p73/env_lock/git-diff-isaaclab_p73.patch || true

# 1) IsaacLab: clone + checkout exact commit
cd /home/piene/p73
git clone https://github.com/isaac-sim/IsaacLab.git
cd /home/piene/p73/IsaacLab
git checkout 73930be06597075d6e587af1f0a55e0439931f83

# 2) IsaacLab: apply local patch (unified diff)
git apply /home/piene/p73/isaaclab_p73/env_lock/git-diff-IsaacLab.patch
```

#### 2.4 Conda env `p73`: conda explicit로 복원 + pip freeze 기반으로 맞추기

현재 머신에서는 `conda activate p73` 시 `ISAACLAB_PATH`, `ISAAC_PATH` 같은 변수가 자동으로 채워지지 않습니다(activation hook 미사용).
즉, **그냥 `p73`를 활성화하고 `cd /home/piene/p73/IsaacLab`로 이동하여 실행**하는 형태입니다.

1) conda 패키지(엄밀 잠금) 복원:

- 입력: `env_lock/conda-p73-linux-64.explicit.txt`
- 형식(Conda가 파일 헤더에 직접 안내):
  - `# This file may be used to create an environment using:`
  - `# $ conda create --name <env> --file <this file>`
  - `@EXPLICIT`
  - (실제 내용은 `env_lock/conda-p73-linux-64.explicit.txt` 상단 참고)

2) pip 패키지 복원:

- 입력: `env_lock/pip-freeze-p73.txt`
- 주의(사실):
  - `pip freeze`에는 **editable 설치**가 포함될 수 있습니다.
  - 현재 머신에서 `pip show`로 확인한 결과, `isaaclab`/`isaaclab_p73`는 아래 로컬 경로가 editable project location(편집 가능 프로젝트 위치)로 잡혀 있습니다:
    - `isaaclab`: `/home/piene/p73/IsaacLab/source/isaaclab`
    - `isaaclab_p73`: `/home/piene/p73/isaaclab_p73/source/isaaclab_p73`
  - 근거:
    - `env_lock/pip-show-isaaclab.txt`의 `Editable project location: ...`
    - `env_lock/pip-show-isaaclab_p73.txt`의 `Editable project location: ...`

따라서 다른 PC에서도 **동일 경로에 소스를 두고**, 동일하게 editable 설치를 수행해야 동작이 일치합니다.

권장 절차(커맨드 예시, copy-paste):

```bash
# 1) conda env 생성 (EXPLICIT spec 기반)
source ~/.bashrc
CONDA_NO_PLUGINS=1 conda create -y -n p73 --file /home/piene/p73/isaaclab_p73/env_lock/conda-p73-linux-64.explicit.txt

# 2) activate
CONDA_NO_PLUGINS=1 conda activate p73

# 3) pip 패키지 설치 (freeze 기반)
# 주의: 목록이 매우 길 수 있으며, 네트워크/미러 상태에 따라 시간이 걸릴 수 있음
python -m pip install --no-deps -r /home/piene/p73/isaaclab_p73/env_lock/pip-freeze-p73.txt

# 4) 로컬 editable 설치(현재 머신과 동일한 형태로 고정)
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab_assets
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab_contrib
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab_mimic
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab_rl
python -m pip install -e /home/piene/p73/IsaacLab/source/isaaclab_tasks
python -m pip install -e /home/piene/p73/isaaclab_p73/source/isaaclab_p73
```

#### 2.5 EULA prompt(사용자 동의 프롬프트) 방지

`isaacsim`/`omni.kit_app`는 import 시 EULA prompt가 발생할 수 있습니다(현재 머신에서도 관측됨).
학습/실행 커맨드에는 아래처럼 환경 변수를 포함해 **비대화형(non-interactive, 사용자 입력 없이)**으로 동작하도록 합니다:

- `OMNI_KIT_ACCEPT_EULA=YES`

## 실행 예시 (현재 머신과 동일한 형태)

아래는 사용자가 실제 사용한 커맨드 형태입니다(복제 시 템플릿으로 사용).

```bash
source ~/.bashrc && CONDA_NO_PLUGINS=1 conda activate p73 && cd /home/piene/p73/IsaacLab && \
PHYSX_GPU_FOUND_LOST_AGG_PAIRS_POW=24 OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
--task=P73-Flat --headless \
--max_iterations=50000 --num_envs=4096 --experiment_name=p73_Flat
```

## 빠른 검증 체크리스트

- [ ] `conda activate p73`가 정상 동작
- [ ] `/home/piene/p73/IsaacLab/_isaac_sim`이 `/home/piene/p73/isaacsim`을 가리킴(symlink)
- [ ] `pip freeze`에서 `isaacsim==5.0.0.0` 및 `isaacsim-rl==5.0.0.0`가 존재
- [ ] `./isaaclab.sh -p ...` 실행 시 EULA prompt가 뜨지 않음 (`OMNI_KIT_ACCEPT_EULA=YES` 포함)
- [ ] `--task=P73-Flat` / `--task=P73-Rough` 가 gym registry(환경 등록) 단계에서 정상 로드

## 메모: 왜 patch/commit까지 맞춰야 하나?

현재 IsaacLab 레포에는 로컬 수정이 존재합니다(근거: `env_lock/git-diff-IsaacLab.patch`).  
따라서 단순히 “IsaacLab v2.3.2 설치”만으로는 동일한 동작이 보장되지 않습니다. **commit(리비전) + patch(패치)**를 같이 고정해야 “완전 동일”에 가까워집니다.


