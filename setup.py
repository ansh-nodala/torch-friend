import os
import sys
import sysconfig
import subprocess
import multiprocessing
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        import torch

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)

        torch_cmake_prefix = os.path.join(
            os.path.dirname(torch.__file__), "share", "cmake"
        )

        # Exact PEP 3149 suffix expected by setuptools (e.g., .cpython-312-x86_64-linux-gnu.so)
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        use_cuda = os.environ.get("USE_CUDA", "OFF")

        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXT_SUFFIX={ext_suffix}",
            f"-DUSE_CUDA={use_cuda}",
            "-DBUILD_EXAMPLES=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(
            ["cmake", "--build", ".", "-j", str(multiprocessing.cpu_count())],
            cwd=build_temp,
            check=True,
        )


readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="torch-friends",
    version="2.0.0",
    author="Ansh Raj",
    author_email="anshraj0000000001@gmail.com",
    description="Ultra-optimized C++/LibTorch Rotary Position Embedding (RoPE) for CPU and CUDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ANSH-ind",
    packages=find_packages(include=["torch_tools", "torch_tools.*"]),
    package_data={"torch_tools": ["*.pyi", "*.so"]},
    ext_modules=[CMakeExtension("torch_tools.torch_tools_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
