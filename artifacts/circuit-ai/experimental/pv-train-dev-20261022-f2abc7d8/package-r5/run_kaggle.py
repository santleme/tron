"""Kaggle script entrypoint; keeps Kaggle's preinstalled CUDA PyTorch intact."""
from __future__ import annotations

import importlib.util
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


SOURCE_ARCHIVE_B64 = "UEsDBBQAAAAIAAAAIQDpU+y3WgEAAJ8CAAAPAAAAcnVuLWNvbmZpZy5qc29ujVLbbusgEHzPV0R+Pk4B2xifn7GWZam3crAFJL2p/16cJoqUo0rnDc0Ms7f53O331brMjO/jGeYTjTkCh+rv/rMwhbOQcRoTf1DBVKf//MAOMiTKI72tFPlIIY/siqJaz7XjlHmea0fnWgmlpVCq9gos9s5UDwae5825ojc4rjOlQ1g/HjVHCOwp3cU34PCSlvCrOk1QGt70UlHboETd6qHvxWAcWNkMRhvju0LIFkQ3COMH6Tu0jbVS694rqbu2hU49lrg7G28kFguh+l5qINtINK7xWnSmGDfKkiYLejADEljwvQcJiEiq7U1v/3FeThEvM66QEGZc+GgjPCFHPHGugev/3TCtC06pWGlxRSZ2jsLtmFKZKz4TxMDheYyQN0YchBDNlUxE22FvZa5oyQqXnnkJo4+A2+PyUd75kqVX4ucpb7UOouBfG1nFU7hn5ZK2X+bYfe2+AVBLAwQUAAAACAAAACEAb2T5CdICAAD4BQAAFAAAAHNvdXJjZS1tYW5pZmVzdC5qc29uzVRdb+Q2DHzPr1jsc5zVF0Wqf6UoFpREJW689lbrTRsc7r+X6zTNFchd+9gXw5JoejScmS93u92+jZNc9j/tftbFbvdle+r2mdcn3d0/cJeZh8ty7UWGLi/jZVzm/f173eWJHcRbpa+WGlKLxaHPgXKIlClQqykVa0xtlsGE5KBgMKkBUXPeFImmxBrzfmv59f5zHM/SZ5mGk6xceeWHXy+fo5BiTDKMHHxLFbKxgSs0xGCTj9UjA2KMIQh6ytW1IFGsBX1Jipl+jKJf56EscxsfvwvAeWtKteIJAzhnJFfPtUUhW5Cit9FVDD6ia1YgcCiMJQJAE2PIpB8DWDuP8zg/Hs7n5XDuy7qUZXo4v36GhIOIgSSYPEFwDKHZlmzLXGNIvgrkXIpOxKJnxuQSkPUGkZylUPA/Iuny23XscpJ5vQzP/Pg4ycP6x/oZolAIrXHVueDFttY4J2RTUw6mWnQGiRG9Y5YIltlJLQgR2aZIYP5lOP9jRBfhXp4O2/p4XqaxvB5feLrKd0bniwOVp0WrUvKVsWKs2cVAVENxzYtAamo4UvVgNXqHnMipxGzzWMNfsPT5y/1m8qWfeL11LmMv13EdeHz39InnscllPdgNyf5xXI9lOZ3GrT7kpP6EoCAqVGUnNw/gIRFSogCxqsAI6OPb35f+vHaRYx37+qo91n6V7fTMRachx3wdpypdj94Y++BrWabL4W1ihzKNH+x8ww1kCdagx+ZLSZiiLaEG5QdENd2wxmazulktbXRkJZlsnOJVcUNIsuXMNrCbm2fpx1VO54lXOX78Q0OpQdLrCmDKKgNPymu0jg21UjwDh+CxWIwcNHBUKRo8YrMjY7HAGxtvBB/fCf6mf86NsNVbPKj6nLXiWJFXb6JGpgHCiF6jsyU9a400LzKxUdNqsSriH/3/DmVtfH4ZNpENVV4GZ1y8tR+a41yw0tBhf/f17k9QSwMEFAAAAAgAAAAhAE81BT3HBgAAkxQAABgAAAB0cmFpbmluZy9wcG8vcHJvdG9jb2wucHmtWEtv2zgQvvtXsNrDShtZtYO+4o0LFN32UKDZAOnuJQ0E2qJsNhQlkFSbtNv/vjMkJeuVNAWqgy2J8/hmOC8qCIK/WMVkxuT2dp4rxghVhud0a8ieiYopTfSeKpaRzS0xinLJ5Y5QmRH2mYqaGl7KJAiCWa7KgqRpXptasTQlvKhKZYBSlsZS6dnMv9tTvRd80zx+0qVs7gtq9s19Lfm2zFhGDXXSK1gEvkb0OTzG5BzUnZea3+CjozO3FYL0ZK/k7Ww2y1hOtlSWIJOKFFWGiJ+tcD0mGd8xbdKcM5GtiDaK/EfOSsnI2v5FZP4SHGCYXs0IXGDwBVOcCv6VkX8+vJ2/IO8u/j4jX7jZEw1awV/X7FZbR715/f7VxVbxyjx+9/qCyLrYMEV0xYQAmNZ5KNMi3LPtdcgNKywuqxb1O6148ZxwzaU2VG6ZpYwRbnSg8FQd7yWyVIXFGgZnb18HMUG+iDxa25s+K16wzZqRf9E/b5QqVRiA4+bA6qwEfQA8iFo+JqZgZXxrBrjyUqFfYjCUi4xwaQEk+KPDaAwExEL4dEVb7rHBfeQfbqsGuAVcbj4xCGi7I0WtDdkwb4TuWNFcbhOA+K4lC/5H1oeCaxMTU1eCRRNu6HlgbMxPaMpFSYeO9o7DdEq4ziFrjSOf8Nv0bjset+EuZAeuQih283TKpWE7psLIxjvdaKeKvCQni8Xz5cnJ8dMnz58sTk6WD9Hupc1tfmZdBKSsjeYZI0iiSqhOdCNgK2nOQI7csVFMAgxwmEtlcDpUBu8G9Hy4KUsRo7A7ndhl0Ej5Syy0Jnkzf84mzfryh/GeB7XUdVW5EmT1+DL37WBKkqaSFlCkv4Ps2SHaLKXTlpXbumBg7tqmcXcJvNKrli2ehiepyirsksSuhs7aMudyLzTsxthia+sc/B9kKQZtRNrWkGR1UWlLHBMG1kF7oXrL+fotBX9ANWAVVdSUSq/DIIbqFqyCqKPNxU5T7t1GjxT+Rs5vzb6Uv2vQXSncbQqdD/2ozVyVtcwAdGUXmQYrbU+LCYWo0B0ph2JPzqze1cqUF9beP0lbh0kpBbTTPeOK7BQtCqoemz0I3pci60jLeJ4zBb2ZaRJiI90A2zJZxGS+wF92U4FnYZu+QujMK5ploCe2Mbpk82ePl+x4GSXd5mG9QNZrshgEknN4sAja15LtwMjP2AMd1ylZtIu4HbCA7ggxHVyARIkov2Ah6KoMWIDZZne7p7Og0nCt6cGO1IvFv0RXgpsQuGOy7Jee1uw1Zk3Y474vW8YKva64YxkiTizihnxQFVzeQslQMCVBEICMhrLBnIwxQ0Jwox1im/dHrYA+HdtyCJK0gpnGSxdMhp4rArYG+z2GtsoaYA9R0dBO6hCMYmx5UicfsqjzlAhM6yqEEIoOprdIxkRYkLvxNoI1Xzdq7xSn7hZX0B10sDrD+D1EaHebMUXI6bpDeUowY0ZdYITsdJQ/eCmco3FUXgMQCKEjxEP+IOF8KABd7Czox7WYUvay5/N71Xr3tIrHLu2KGiTVKIomhV+uhkKvUJ8111MMCVZX94QqVHKIM57zbdpJygngy/4IdUg+r3XZUTONfIFIQwe15QePt/eIDmb7aFrO0dqWMhRxFCDfFHTYrEUrB2ghPsMJuoMKX3jDYG5ltkW3I6NBcGhp0BFgth8cFHr9bDD8TBd7WQsRTLF8UPUdLAZWJllsM57myXGpxzQaqyYZ0XeW6G5W280nmX3XH/E/4PR0UI9Tyo8l3DfqNz64tOUgDpJPJZeh3z+YW2AUwdMA3jWHAdzy4Cq4V+XE2arR9K2nyRuBpxl8vcLFTvRcwsLV+JwzdfmzG6J0J1znGSv2+wHtL5pIvTUeajNaRol/EdQmn79AYnuy99Nm91iPHroE62PMj6uJE34/Z7w+/2ki0Xt6/PRZOPXFoC8KRp49u3FvwgZOzgVLvQj8aNF8UcBPFH21jhFK1ECx2xL7OQGZrJQogdOBDAO1gaoAc+cexjzRSTl3qqzltQ8kFQpabDK68pSJgmYaLhfHT6A14F8Ukw1UmNHoAIiSusooHBmtvKjrIb8+NhpPLKliwpavFAE3WzFyNi6Czb1PN4MjBp5e3dRps8MeY1No46WoAZZr9h8/2jGtJQs+3iwWg1eJG+WsBEwyvRpEaudcZkMVz13tJzC3ed+suEfqe9DzhH3rjf+iwOEuSIYbHpOHRuTEN6fWZYaqHcNAOYRD5z2ahiev4jrjKnQPeo1VHEddqE5peW0ff3CEs0gvu2tXrn32s2uQAl0gzhH2U9lk9kQ9F3YTxQmIZv8DUEsDBBQAAAAIAAAAIQAvhgDG+gAAAGIBAAAgAAAAdHJhaW5pbmcvcmVxdWlyZW1lbnRzLWthZ2dsZS50eHQtUMFOwzAMvfcrLO3AZavWaUMI1koIblwQgvPkJl4bSJ0oTlH797hlNz/7+b1nb+ANu84TuAE7AkuR2BIbR/IIjjNxdoHR+xloMn60JJBDMj1IgOgiGGQOGRJFj4aKzU3vTiAmcixZd8nCy9fr844Y2wW8z5+rRDs6b0v4IBOSBTR5RA+/lEQtRd1VjVB5aWQYkN2VJD/BTRQCa6jckxDIGGNI2XEHEc2PXqLrV8BWNH9ZBOapqauyetieDysSk1zMTb0vT9tztbZ2Li1YWfcLa/AXm+dIspKKf1k1aOrDsdCBlhea9D9r2KY+ltW++BatTU8Drg21OxV/UEsDBBQAAAAIAAAAIQD5kixapxYAAHFOAAAlAAAAdHJhaW5pbmcvc2VhcmNoL3RyYWluX3BvbGljeV92YWx1ZS5wed08/XPbxnK/66+44k0rwAZgUVEclxNm6vo5TaaJ44ndtPNYDgKBRxLPIIAAoCRakf/27scdcAeAFJ3Oe52pfrBJ3O3e3t5+74GO47zblbK6SWu5FKv0Ti6DpSybjXhbZGmyD36Js50Uy7Ru0iyLm7TIxaqoRC3jKtkIeQew6VbmTR2enX2fl7tGLOMmrmVTi10tRbORIim2ZZw0Iq6qeF+L2yptGpmL67349demitM8zdchIwzVQr/+Gp69V0MirUWaAwQuHmfZHhYv4ypupFhVxZaWKFarNEnjTLx9+5Mo01JmaS7DM8dxzmhOFK12za6SUSTSbVlUQEyeFw3tpz4708+qNSCuJcPgPpIsrmtZt0D1Mk0avxvSkH+ti1x/3sbNhjGU8ClLrzX0WxxQn6s4Xxbb9ptastmXtGF++jLfn6nnmktlWYRlVTRFUmR62jJdy7qJkAQfTjCTUb2JL798fnb207++e/3zLy/ff//Tm+jdq+9e//hSzISTJUFxXcOR0+6fTZyzl6+GU+DA9OjPr76L3rz88TWObLMy2NwmwQpkAY5Ejf7y+ud3gAEnXDpnZ2f/0jLIBfo/ynz2vtpJ74weCTrYV0W+StfTMwF/siySTT3FUwYUlxf08Dpukk1Upx9lOwB7wpFNulzK3BqaXL6goQykCBkVoXhMxSorYhz+QgZXNH6D0hzdynS9abrhSXihR9MlcSVaVcyAbtJFOKFJtZTLjtTL55OLy0saWMqbNIFF66ZCRsS7pgBe8MhK45ZuLbOVJ4JvxJsil7x//EOlyuOtBMzCdZgjji+cjg34zdg6fkVaHK9D0m4RCFhLEO+movV8Qu1Z89IVipt0ab6HSgYKQfsCShjJ18K9wHlE12ym1hMyA8We9JbFPzhXGCGL8bqqispdOfcI+yC2u7oR11LE4v48L/Igl2vg8408N9GfI/pzRn9eFnVKMx5I+deycjr6EahgTQvTegW6oRgbWgLgCbJU/cfiazhNm/oB5Y4NoenXVIH5QGuJ655Elil3HVXmU2D2ozRZ8zVJJjv/AFl9gSfqcPoFsunQLKB2cgK1AzBNNAj5/MIHIbLJpOVYj5Q0gi6QIoGwJ+WO/tst477MDxdXSFq5Axy+AAw+bo9RgJ1CxQT1XkbKZ7losVmDfydzTYqKNn8Oz3y0yAteGTzLDwBIDNeqTZ5IOTBQv1yC6BTVeU0OsJI1etg3b/8CX/MGGRKie0JkTbXv9qOser7blntwOCIv2UbeJeCWxfc0SpvEQXg6FeJPoqzi9TaeAtMA+42sRCD+PV6vM5YI7T6EzG9qmFvcpEvJK5zZLPx5B152q5nINIBlqORvu7QC8puC+NULDpTDdzx2yEAU273bFOKIvAwRxEV2En89X4AXL26jMk0+ZHL2bQwK7+FuMAIAKe540a47E/eO4bTIMpZIQyrpM2kGfcpAFbJoG9cfam0ha+ehrwsaMehDvbvGk1drh+hA68fEa+WoPSNztoADmcvhzVTc13BCcum2xAdiuID3YEg+IsMtoiGcai7M8dvC8gsa4wOz12SILzQ7ABHim9v8Wvjqacu1RecQAZoY1oIqdrZAJlMZDvho4g+X6EvEP8zwuHdgsF+gnlkz8mW6xQlXgxEIV0o5DyYLHJ5cTs8O8t3aU6vcvN78jf+d/5/+5HKhOJsUIMy+2JC59EEYl83GFxFscri63hPBgHG7RCI32jDTN4LHL8fo02KRo9gJiBvAjQCRzW0BWhFvSzh6AWbwuogrc/zy7lIRDTToE7J4SlHIF0RIO06U47irtnrlHaOtFRDNN40TOHe16Nbnwz+0uhrtr310ZYYZWddYlYRLLapNP6xMD/DDdVFkEZgO8xkdvEcei8E/kyOGWLfE4TrMEeGyYF08m6Cag5UqRJygFZZL7bfIvLRqw8amVRD6avER8D2/YvePQwMm4pDLOCEc8MI437tHd8BzR0MBXkuxniwxL9dxnbYPxDPvQA6Rrbh/8JRFuVeGWe8GD4UgiKr4Lq1nE6Awy45TKMEb7Q0PxFrAJG9iDFm0EkA4rEjiYMGxlkbu6dhFS7JanrjWSvdpjLOVAT0yEKhCJ5uP4C6v4+s0g7gPTOJBmpjRJkUwGl/XekB8IyYn0NVXFSYInTiGTMHEnxg60256/okYt8BFZPDVqfuHc4GUF1N1SMvSdW7sdY+uPs4FhDLHjgR2m2RF3R1JWO+2Wjh8TKx8UTVFNgOyvgS/rz4+Py4ycbIRFoHMD0CNVIGcKDpGnBOIM30/4/ABcv6cpqlgL1pXxa6UywgVJ6pL2Cnrm38k+fO7nM9ro793CNsZ9es9ZLFpXUBcVRXAG7QVy7/GiQR3UkMiBlNyVAWRVAWkwBgpXoPOL+Nq34aBI5EfPd/l6W+U0qHJo89Ms3UYFK4/Gqn/76J0Yy7xj70lJFOG6xgg+wZIQ/nNZO4y9aALM9ONHkK7TXPXhArExB/knGN/2/jOnfh4Xm6FTLawPBmj0oM/Nuib3WqVUcQJ3OYqDWQkq3iXNVGVr4nzXljKarvj8pFGPMYhdA0a43w6us9FHw4FmVdPIawckU7j7I2nKcRXCUV/5K/jBozYR1kVbg+1pzIOtseHwT6NwrV+IHf7KB71UwHpniClQ0uz3CUSEyght2VjuAiIn5S8Kf3tr+SP7Fvr9xY0MIu49ATJV7LpNPZVJXWSdpPKW1kFVJhs0mvwR8kmznOZBRl6IlXcAup+/OGtIJStknIJi/Mf2iEvE+Z5+CNsKZMGH4iiCE14FKk6DEWRZEp0ZKq/9IpavRSkxiKt64UtOruYQ4mzzBMgFYtPLUnvIGPAwmmcuQPNaSd9y9t1vaF2tXN+SHMZVy4HwU90aPwEwnWLdLD5Lcj7ON+cgtMA/yxcIxxQTmN2fI2rEUhdNTsKOPHOrLOF3OwWAnl1tEZW0Tu8FQjeriItMw/KNSEsgKxYp007nXflaiz21D+JH7FCeECstUFHm3JziRX5JboqnIyVchDvaQ9bhV4KtBeL7uz7fTLft+Dq62eoihkm6hKz/Gy3BHy3G5nDlL2o45XsYVMc0nRICAyaOhTvYX2Zr4HFsF5dZDfAmzrdgo2Nc1nsapCveNlDlRRZBhYRE7941YCYXxcgilUKjKxBfQuIQkILBHFEZQYY7ZRvHoahLy4Wg8kwrYMBT1uD7siP0g0m3mAumiOaV48hv5q+GKJH2wahfg/6CY15FDdBhjxDx3XpQQRXrdGN4aP++vqIDGld7fKE2yMhhEfRpmhcvZB4CvjEP4orHyOLSHU0Zlde2BTKdmHM39+ikTTbO1Tp+mT6hb3FO705xfMniroYnOhauir95rrczMTHj07z7PqPkiobCz7xQlQBZOFE0e4xZzVXswJcuL3V/XG6NSf+XoTr9SaPU14kya5MKVZxByJ4uYCDHzz9YuGFoHF4giDWfmfHmVkYrNlLrFbUytOiBm6iBn/uBuo4J21dJcB4/yCTbMKXsm7SnJuJM7GfT31qhiw0HUD5XffwqaZijt99MbXlLs0jiqQ7IiHeTj647h5344Ncfq2wYvwIx/213jV9v8NZSDoo2uQgmZHBa14EssMNmPDB4euJxBsDQwiKty3diz7PkQrPXhcsD1hSWOlTt7ffR6mxdTauIwSEGOgGod1PCtE/iU/KZHBiSFv1xQcpS/zIrTmLpZkuDWhT87twW2TWOgf8Fn8I0e9AoLsChG7rTZh/kNmuClfNUzoAAX/f0lEAyLN8w1N3zjDsW2kFY0RobWzI1LA7jbiZAsmrChT9dpmykljyh6m6OspZaKST3MdUXk3X0EaC6o91IFW0oMg7xsNPFqoTGKn2irsEjIHb20hr7AA8qotVgy7HooMVI9DWSH0JtzLO3TYHUXw84JC2tWLzQb6qEol5fCbhTy2mcQKnlvTNib4xoM882cjkQ1lAQB2lS7fN431R7SB3WOpeD1oWIB//o5YPPOVDSSD24ebOTKyce4Z6CHjVgNZzMCviAe5drhy+zBCUNwElPvf474NjZlCVBBZl2RZjN7dy5i+Dv8TBx4vgnxfdxzCaBov7C39y+dWD43ekHM20FCGUwqskS8SiY4P4/s9cDSzybA+iCZF/RafMp30ensM/Ef4zPadm2XlwbqdjLR2Kx5SeuaryHfU7aIBj15S7JlqmlfXc0KGE7wKYFwOA3cY3CP3FEx9vmxRVExV5fjflQi3O2lku+OC5+rpPBYK+q3Sjfnj0Y90+JlA3QqXblZ9IT40GoskGnWo/3twzhkiD/ljHrwEPIalQqHp/FXfzjvb4rE5em4OrTlOtKYUE4O3+PZJmNvmUPCvuqJYr3hOgBitBoayzUcBnYVpH8Q2kEDEkJ8MCpU3Zq//480sihLd1vWvAM7bQugSoFlWL8FdXEXCEtqI6YsPdIRQ1obmZemg3Hms/9amVwnD5CLVfo6TyEQ125aXxcV5nG+c7sPjjU2B/TGJIPYZ2dx1fDWINTBEWqQ8sCClitJRgFMCHYCs7gcnrogKLsa1dUjZIBSsIQcB8GKGDirxA8jHJaDYVBNS1qwpxWMi70v5q3ZtDpbdBP/NAF/OPNfZOKSAB9Fhx2OrRth0ev2cRzFJiO9Zx1pila5tYQRsSwdO5hsVHOlbR6iZhkbSdrFMszRdFhllRwZTPDMiLEvQNBrqaET0JXy7jrct1L7xpt0V5qNEIZ9VMobUv2vABcnaArLxv+WYf4FQnV3WkUgm7ic3pkQpsuE3YZhWK8M7Yd73sIdp+s/tzUasO+CHERoP8s9CavYpDuK1+hr0AN+fGsPMNh/bSSXue/6afqHOfsXk6blc2oPgFuC2RwYd55xBpfwvsscxZE/FyAl1Vo9sJlC8rZHyBzbDwLE0cK3RBvlFuZ3rRJGJx3R0tMfvdDmftJy8kJ2VgpcUpDJS12gST3lGuqYdcsWo66i/88dp2q03ddbxeja8zI33gedsBoMWmvOTTIcaFndjSQMuYTkpaI2ZKBpYH+pIxkpoNkxttQlylvH2LOycyFn1kGG1HbHQP51Xjpd+jiZYmotNcRcBIHVnP1cqoZ3Yjliq1w53t7u7k2VUPbRZDbIWAW4gxdWiipohyCBNHMmbcLpxk8oHqwYewQRxT9gZNUQ3jspT50iVZdQkleOIYEgQvBK11PaNOoFRUg9zzjVFHZZQuK+VTgUUZhwSS1mjHIfigFM5c3vMelB1nXZXAIEUtJQwqt1P8MMTfbvyRh5ueJMlDB/h/KdQA2qWRUT+vjKw89/+fzHdnQZftLGnRnFFSqO/e7aPBLOO5nnwkzlYXWre1tJB03OwW1I12h9J3ty9vXnfDD4PwwY3oka1xTtjbiXpo0qUeGQRc6Pt3lNgCTrra2KW56vpZm3JTi4DnPoMIvXvu9CeG2w8A70LMhW9UzDjcpqZJVHwwzI5V1UBxtMschkPXhQ5VDuLEFCBM6p5hZcPC8BCWDROHufapMDjXMbOB+EaCZQI3u40b4Bu+VtCBPJvQbV4TBcyxvg9lx0ETitPMKGNkmqHuUQ04t5AUieHrEEAAB+3dJOtdiBHMHG/THc1GJliDBKB7B69oIrR+VwKFCGJmvDOjnqp3JE5uFzi8R8JOkT18UhG+wyE+PFCx/me1IBR681WC6Uiy8PAwsvs2tmmJ41dilMihpjKD6M5LhEMwh/1J92isK+y0btKGNb1nB//ga1lWLgt8kFkd6hJgs/pCE7nMgrOwjJIXv8VT8e3VxaQzIAfrLnZteqRk8dObN/+l6OhKKNhs1Xx7pp7SG1PBByrThM1dA04O/bjMyf73Siz4V/O1Oe1DMS6pXbNfpBsyl95JOUmvQoDsCJlwzvt8taKPVTK3tQIe3qpBM4fyXs8sv+qM+KvuT5lHDcfusL26jYlOUWKMpbRmNvnqGLblHhBhbeIOsFlXw1FfLqbqvRnnAW+E81K95yerjCLQBge8REJhXpKkbSKnIF+O9Z1uB+97o3szDf+z+5ajIb0gQ1fV6a0tnNq9w2Vy/jDFDrEOIIFpwlFVPzS4CK2+Brfy2rgHX+7p4E8hVSnaI4RqdTzOWMMZdOr8tNVxg0BN2LRXkLWKCz2Un+1fTvEnfx+L/ze09p9t6f823pOEEQ26qSEYgLEkwkhfJnvwkDWnK1gQVXluhhWEGDAp8bUWmPPTo6bJkOgerHq+OGouTFL0Zjpq+pv6TIKG4C1NBoqh2swd/SnSAFhS7V4VdY0j6E/1BuhODQH1/BCXcMbRhPgCsIQc8K5xcVq43G3L2qAH35+JPsi9DoDV676Q3cxcx0feTh3Pg+zW+e/cOUEb6LoVON+Zs2tWwQu1PTtP6G8Uo/6ztsujOzo61h92edgrbuM8XQGHVfVaTwk5oMfoX88wGGSDRpu4xnVM22rP8DDYsR9hJwIBXNWDIOpN4tMl3gJs9mTmuf8FwmU3xLDZZ1GNFu4we8fdgMWZI8DtdlssI2x4UG1HiqdmopdK6Car0YoN2jiLjD+nPjBdNUXx1b9j+cWj3kLtzmCdZuwAlckIh3qNx3ljJpiqwdFF273sd7REOcTYTuluGxsIusvkVmtj3ke88EZQj15lPgH5sOLzCHrA2X3xDTc5yDs4sNWs7lfeFWJVNkNfyp+sEEEBHwpltAezI+Je6tEp4ABc2xiFwjI5hMY2Qi2iB6PacGqFgJVGG62u+qBPN+AJhhkyIEwL3W7CMNU8dWioU8xgmtmlfsEzj3P9dqcy1gdssUoO6NVIrNGrE2jfpBpN7UwgyF0arIN25+Lb3PSGC+naqbFxxbtd3tqho+Hn4/YIUH2uKTrJyFAQoQIhZWHo4CqZ0VtLUVOogpSH9VZ8bf0O+3cHYnfc+sOIscYOCYlQmlusW5hNJ5Nd84Fh74cd5mx/6Af4BNxOYG2H6T0qmhb6zxfQrgowKqgsK90uaXtkDw4RrMDo5gxDm9dmIkz0o7z8eOT+zNjtmN97YjnqaQ9cvMGbNMYtmVHQU27atH9//F4NvkfV4O0JvKq0ruItaFOaAO+bai/4uhKKIFZQ1JWW7gdn4gx3kcuqe38KwqI6zSGxzBPdiPRpXaNZkYxdL3ryRPkRjYfeQB3iMmAGl1fe70tdBlJr6NeozIPDYpLYguXBWzbFyhpbpTJb1iPvvljCYV6o0sJhXY2aGZ9HzouPfsb/9S9Fzeyv+jcStkhEXK1vVDcVjnExcsSpbvzQr+hgE1r/ok74slrv8NTe0gj45zqp0hKd+iyKlkUSRZ4BGcbLJWTcDNK+VQ12lApbLMUbmZUzx/hpBes3inTEFIJuOUdQBwFz08atX7Q3HOoB6PaHWgiabhaqt8VmlxdHIamSFKhfchmB/vL5UXDO3w/DTy5fHIXX9zcCvL+hMajXHTUO/Nmco0jYyXE76QCOSXicDV10F+gbNAcwXYSTo5jox2nGD4J/pecotJKxQCVHx+dyqOnjaycYw84O/EhJRwKPH0WaF4EqZDAXZg4GqODqQAQVJEzH8EghoP8QRU2qqX/0YMS8sZDOcKoucRk/r8TPu+9HMmqjaMRQ1utR1pUgHrceHUFs9iUZ0nxyHLB/AauFH97MQhHhYeqNHUariuU01ayYlxVmOEa8wfaZ57EE+XRMIZuVcQON3oUm5cUhK22QYhtocyltpQ/AH4p+PMvDXICNT/E9RUz3o4huEUYRWvwocg5dNZV3skroJ+PwZ1vEqx++766dMnLyie/2dSO3r+/SxiUXAiv/D1BLAQIUAxQAAAAIAAAAIQDpU+y3WgEAAJ8CAAAPAAAAAAAAAAAAAACkgQAAAABydW4tY29uZmlnLmpzb25QSwECFAMUAAAACAAAACEAb2T5CdICAAD4BQAAFAAAAAAAAAAAAAAApIGHAQAAc291cmNlLW1hbmlmZXN0Lmpzb25QSwECFAMUAAAACAAAACEATzUFPccGAACTFAAAGAAAAAAAAAAAAAAApIGLBAAAdHJhaW5pbmcvcHBvL3Byb3RvY29sLnB5UEsBAhQDFAAAAAgAAAAhAC+GAMb6AAAAYgEAACAAAAAAAAAAAAAAAKSBiAsAAHRyYWluaW5nL3JlcXVpcmVtZW50cy1rYWdnbGUudHh0UEsBAhQDFAAAAAgAAAAhAPmSLFqnFgAAcU4AACUAAAAAAAAAAAAAAKSBwAwAAHRyYWluaW5nL3NlYXJjaC90cmFpbl9wb2xpY3lfdmFsdWUucHlQSwUGAAAAAAUABQBmAQAAqiMAAAAA"
SOURCE_TMP = tempfile.TemporaryDirectory(prefix="circuit-ai-source-")
ROOT = Path(SOURCE_TMP.name)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    for item in archive.infolist():
        normalized = item.filename.replace(chr(92), "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise RuntimeError("Embedded source bundle contains an unsafe path.")
    archive.extractall(ROOT)
sys.path.insert(0, str(ROOT))


def ensure_support_dependencies(torch):
    missing = [name for name in ("onnx", "onnxscript", "onnx_ir", "ml_dtypes", "jsonschema")
               if importlib.util.find_spec(name) is None]
    if not missing:
        return
    # This requirements file intentionally does not list torch. --no-deps
    # prevents pip from replacing the image's CUDA torch or its dependencies.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--no-deps", "-r", str(ROOT / "training" / "requirements-kaggle.txt"),
    ])
    after = __import__("torch")
    if (after.__version__, after.version.cuda) != (torch.__version__, torch.version.cuda):
        raise RuntimeError("PyTorch/CUDA changed while installing support packages; refusing to train.")
    still_missing = [name for name in missing if importlib.util.find_spec(name) is None]
    if still_missing:
        raise RuntimeError(f"Kaggle image lacks support packages {still_missing}; dependencies were not resolved.")


def constrain_cuda_visibility():
    """Expose one deterministic Kaggle GPU before importing CUDA libraries."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return
    first_device = configured.split(",", 1)[0].strip()
    if not first_device:
        raise RuntimeError("Kaggle benchmark requires at least one visible CUDA device.")
    os.environ["CUDA_VISIBLE_DEVICES"] = first_device


def main():
    settings = json.loads((ROOT / "run-config.json").read_text(encoding="utf-8"))
    if "simultaneous_search" in settings:
        run_id = settings.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Experimental search run config must contain run_id")
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        source_manifest_sha256 = json.loads(
            (ROOT / "source-manifest.json").read_text(encoding="utf-8")
        )["source_manifest_sha256"]
        from training.search.tournament import run_experiment
        run_experiment(settings["simultaneous_search"], output,
                       source_manifest_sha256=source_manifest_sha256)
        return
    if "policy_value_data" in settings:
        import signal
        from training.search.distill import DistillConfig, write_dataset
        config = settings["policy_value_data"]
        run_id = settings["run_id"]
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        dataset_config = DistillConfig(
            seeds=tuple(config["seeds"]), width=config["width"], height=config["height"],
            edge_budget=config["edge_budget"], max_depth=config["max_depth"],
            heuristic_styles=tuple(config["heuristic_styles"]))
        class WorkLimit(TimeoutError):
            pass
        def expired(_signum, _frame):
            raise WorkLimit("Policy-Value teacher dataset exceeded its configured wall-time budget")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(config["max_wall_seconds"])
        try:
            manifest = write_dataset(dataset_config, output, format="npz",
                                     experiment_id=config["experiment_id"])
        finally:
            signal.alarm(0)
        manifest_path = output / "manifest.json"
        manifest["run_id"] = run_id
        manifest["kaggle_runtime"] = {
            "gpu_enabled": False, "internet_enabled": False,
            "python_version": sys.version,
            "source_manifest_sha256": json.loads(
                (ROOT / "source-manifest.json").read_text(encoding="utf-8"))["source_manifest_sha256"],
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False).encode()).hexdigest(),
        }
        manifest.pop("manifest_sha256", None)
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                            ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
        print(f"Completed CPU-only Policy-Value dataset {run_id}; examples={manifest['sample_count']}")
        return
    if "policy_value_train" in settings:
        config = settings["policy_value_train"]
        source = config["dataset_source"]
        input_root = Path("/kaggle/input")
        dataset_owner, dataset_slug = source.split("/", 1)
        dataset_root = input_root / dataset_slug
        from training.ppo.protocol import file_sha256
        mount_entries = []
        if input_root.is_dir():
            mount_entries = sorted(entry.name for entry in input_root.iterdir())
        mount_roots = [dataset_root, input_root / dataset_owner / dataset_slug,
                       input_root / "datasets" / dataset_owner / dataset_slug,
                       input_root / "datasets" / dataset_slug]
        mount_roots.extend(input_root / name for name in mount_entries if name != dataset_slug)
        if input_root.is_dir():
            mount_roots.extend(path.parent for path in input_root.rglob(
                config["dataset_manifest_file"]))
        mount_roots = list(dict.fromkeys(mount_roots))
        resolved_pair = None
        for candidate_root in mount_roots:
            try:
                mounted_root = candidate_root.resolve(strict=True)
                candidate_manifest = candidate_root / config["dataset_manifest_file"]
                candidate_data = candidate_root / config["dataset_file"]
                mounted_manifest = candidate_manifest.resolve(strict=True)
                mounted_data = candidate_data.resolve(strict=True)
            except OSError:
                continue
            if (mounted_manifest.is_relative_to(mounted_root)
                    and mounted_data.is_relative_to(mounted_root)
                    and mounted_manifest.is_file() and mounted_data.is_file()
                    and file_sha256(mounted_manifest) == config["dataset_manifest_sha256"]
                    and file_sha256(mounted_data) == config["dataset_sha256"]):
                resolved_pair = (candidate_manifest, candidate_data)
                break
        if resolved_pair is None:
            raise RuntimeError(
                "No mounted Policy-Value dataset matched its pinned hashes; "
                f"source={source!r}, input_entries={mount_entries!r}.")
        manifest_path, dataset_path = resolved_pair
        try:
            dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Mounted Policy-Value dataset manifest is invalid.") from exc
        digest_document = dict(dataset_manifest)
        recorded_manifest_sha = digest_document.pop("manifest_sha256", None)
        canonical_manifest_sha = hashlib.sha256(json.dumps(
            digest_document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()
        data_record = dataset_manifest.get("data", {})
        dataset_config = dataset_manifest.get("config", {})
        dataset_sample_count = dataset_manifest.get("sample_count")
        seeds = dataset_config.get("seeds") if isinstance(dataset_config, dict) else None
        protected_seed_ranges = ((0, 31), (1_000_000, 1_000_255),
                                 (2_000_000, 2_999_999), (3_000_000, 3_999_999),
                                 (4_000_000, 4_000_255), (5_000_000, 5_000_271))
        styles = dataset_config.get("heuristic_styles") if isinstance(dataset_config, dict) else None
        allowed_styles = {"cruise", "local-mobility", "interceptor"}
        width = dataset_config.get("width") if isinstance(dataset_config, dict) else None
        height = dataset_config.get("height") if isinstance(dataset_config, dict) else None
        edge_budget = dataset_config.get("edge_budget") if isinstance(dataset_config, dict) else None
        max_depth = dataset_config.get("max_depth") if isinstance(dataset_config, dict) else None
        sample_bound_ok = (
            type(width) is int and type(height) is int and width >= 2 and height >= 2
            and width <= 32 and height <= 32 and width * height <= 1024
            and type(seeds) is list and 1 <= len(seeds) <= 256
            and all(type(seed) is int and 0 <= seed < (1 << 53) for seed in seeds)
            and len(seeds) == len(set(seeds))
            and not any(lower <= seed <= upper for seed in seeds
                        for lower, upper in protected_seed_ranges)
            and type(edge_budget) is int and 8 <= edge_budget <= 512
            and type(max_depth) is int and 1 <= max_depth <= 12
            and edge_budget >= max_depth + 1
            and isinstance(styles, list) and 1 <= len(styles) <= 3
            and len(styles) == len(set(styles)) and set(styles) <= allowed_styles
            and type(dataset_sample_count) is int and dataset_sample_count > 0
            and dataset_sample_count
                <= len(seeds) * 2 * (width * height + 1) <= 50_000)
        expected_source_paths = {
            "training/arena/engine.py", "training/arena/heuristics.py",
            "training/arena/seeds.py", "training/search/bitboard.py",
            "training/search/matrix.py", "training/search/planners.py",
            "training/search/territory.py", "training/search/native/territory.py",
            "training/search/native/territory.rs", "training/search/distill.py",
        }
        source_hashes = dataset_manifest.get("source_sha256", {})
        if (dataset_manifest.get("format") != "circuit-ai-distillation-dataset/1"
                or dataset_manifest.get("experiment_id") != config["dataset_experiment_id"]
                or dataset_manifest.get("run_id") != config["dataset_experiment_id"]
                or dataset_manifest.get("protection_scope") !=
                   "development seeds only; no official/final/reserved split data accessed"
                or dataset_manifest.get("teacher", {}).get("algorithm") != "fixed-depth"
                or not sample_bound_ok
                or dataset_manifest.get("opponents") != [f"heuristic:{style}" for style in styles]
                or dataset_manifest.get("sample_schema", {}).get("id") != "circuit-ai-policy-value-sample/1"
                or not isinstance(source_hashes, dict) or set(source_hashes) != expected_source_paths
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                       for value in source_hashes.values())
                or recorded_manifest_sha != canonical_manifest_sha
                or data_record.get("file") != config["dataset_file"]
                or data_record.get("sha256") != config["dataset_sha256"]):
            raise RuntimeError("Mounted Policy-Value dataset manifest identity or embedded digest is invalid.")
        if (config["epochs"] *
                ((dataset_sample_count + config["batch_size"] - 1) // config["batch_size"]) > 20_000):
            raise RuntimeError("Policy-Value training exceeds the bounded Kaggle optimizer-step budget.")
        constrain_cuda_visibility()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Policy-Value Kaggle training requires CUDA; refusing CPU fallback.")
        torch.cuda.set_device(0)
        ensure_support_dependencies(torch)
        run_id = settings["run_id"]
        output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
        output = output_root / run_id
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
        output.mkdir(parents=True)
        from training.search.train_policy_value import TrainConfig, train_from_npz
        trainer_config = {key: config[key] for key in (
            "epochs", "batch_size", "hidden_size", "learning_rate", "value_weight",
            "validation_fraction", "seed")}
        trainer_config["device"] = "cuda"
        runtime = {"gpu_enabled": True, "device": torch.cuda.get_device_name(0),
                   "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                   "source_manifest_sha256": json.loads(
                       (ROOT / "source-manifest.json").read_text(encoding="utf-8"))["source_manifest_sha256"]}
        report = train_from_npz(dataset_path, TrainConfig(**trainer_config), output,
                                run_id=run_id, export_onnx=True, dataset_source=source)
        report["dataset"] = {"source": source, "sha256": config["dataset_sha256"],
                             "manifest_sha256": config["dataset_manifest_sha256"]}
        report["runtime"] = runtime
        (output / "training-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset"] = report["dataset"]
        manifest["runtime"] = runtime
        manifest["files"] = [{"path": path.relative_to(output).as_posix(),
                              "sha256": file_sha256(path)}
                             for path in sorted(output.rglob("*"))
                             if path.is_file() and path != manifest_path]
        manifest.pop("manifest_sha256", None)
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2,
                                            ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
        print(f"Completed CUDA Policy-Value training {run_id}; artifacts: {output}")
        return
    # Import/check PyTorch before any optional dependency installation. Kaggle
    # training is deliberately GPU-only; local training retains CPU support.
    # Kaggle can initialize CUDA before this script runs, so changing
    # CUDA_VISIBLE_DEVICES may not affect torch.cuda.device_count(). Select
    # cuda:0 explicitly and record visible devices separately from usage.
    constrain_cuda_visibility()
    import torch
    settings = json.loads((ROOT / "run-config.json").read_text(encoding="utf-8"))
    campaign_id = (settings.get("g008_campaign_train", {}).get("campaign_id")
                   or settings.get("g008_evaluation", {}).get("campaign_id"))
    v2_nonfinal = (campaign_id == "g008-competitive-20261021-v2"
                   and settings.get("g008_evaluation", {}).get("mode") != "final")
    if v2_nonfinal:
        forbidden_sources = (
            ROOT / "training/configs/g008-split-seed-plan.json",
            ROOT / "training/eval/provenance.py",
            ROOT / "training/eval/registry.py",
            ROOT / "training/eval/lc_eval.py",
            ROOT / "training/eval/g008_final_roster.py",
            ROOT / "training/eval/g008_final_opponent.py",
            ROOT / "training/ppo/g008_benchmark.py",
            ROOT / "training/ppo/g008_incumbent",
            ROOT / "schemas/g008-split-seed-plan-3.schema.json",
        )
        if any(path.exists() for path in forbidden_sources):
            raise RuntimeError("Nonfinal G008 campaign package contains final-only source or seed data.")
        if not (ROOT / "training/eval/g008_campaign_seed.py").is_file():
            raise RuntimeError("Nonfinal G008 campaign package is missing its safe seed codec.")
    requested_device = settings.get("train_config", {}).get("device", "auto")
    if requested_device == "cpu":
        raise RuntimeError("Kaggle self-play training requires CUDA; device=cpu is not allowed.")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("Kaggle self-play training requires an available CUDA GPU; refusing CPU fallback.")
    device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    torch.cuda.set_device(0)
    print(f"Preinstalled PyTorch={torch.__version__} CUDA={torch.version.cuda} available={cuda_available} "
          f"visible_cuda_devices={device_names}; training_device={device_names[0] if device_names else 'none'}")
    run_id = settings.get("run_id") or settings.get("g008_evaluation", {}).get("run_uuid")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Kaggle run config must identify run_id or g008_evaluation.run_uuid")
    campaign_runtime = any(key in settings for key in
                           ("g008_principal_benchmark", "g008_train_schedule", "g008_campaign_train",
                            "g008_evaluation"))
    if campaign_runtime and (not device_names or device_names[0] != "Tesla T4"):
        raise RuntimeError(f"Registered G008 workloads require cuda:0 to be a Tesla T4, got {device_names}.")
    runtime_evidence = {
        "format": "circuit-ai-kaggle-runtime-evidence/1", "run_id": run_id,
        "python_version": sys.version, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "visible_cuda_devices": device_names,
        "visible_cuda_device_count": len(device_names),
        "training_device": device_names[0] if device_names else None,
        "training_device_index": 0, "training_cuda_device_count": 1,
        "source_manifest_sha256": json.loads(
            (ROOT / "source-manifest.json").read_text(encoding="utf-8")
        )["source_manifest_sha256"],
    }
    if "train_config" in settings:
        settings["train_config"]["device"] = "cuda"
    output_root = Path(os.environ.get("ARENA_OUTPUT_DIR", "/kaggle/working/arena-output"))
    output = output_root / run_id
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kaggle output: {output}")
    output.mkdir(parents=True)
    g008_settings = settings.get("g008_principal_benchmark")
    if g008_settings is not None:
        from training.ppo.g008_benchmark import run_principal_benchmark
        report = run_principal_benchmark(output, run_ids=g008_settings["run_ids"])
        report_path = output / "g008-principal-benchmark.json"
        from training.ppo.protocol import file_sha256
        evidence = {
            "format": "circuit-ai-kaggle-g008-benchmark-evidence/1",
            "run_id": run_id,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "visible_cuda_devices": device_names,
            "visible_cuda_device_count": len(device_names),
            "training_device": device_names[0] if device_names else None,
            "training_device_index": 0, "training_cuda_device_count": 1,
            "source_manifest_sha256": runtime_evidence["source_manifest_sha256"],
            "benchmark_report_sha256": file_sha256(report_path),
            "benchmark_report": report,
        }
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed G008 principal benchmark {run_id}; artifacts: {output}")
        return
    ensure_support_dependencies(torch)
    evaluation_settings = settings.get("g008_evaluation")
    if evaluation_settings is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Registered G008 evaluation requires the Kaggle CUDA runtime.")
        from training.eval.g008_validation import evaluate_registered
        from training.ppo.protocol import digest_json, file_sha256
        evaluation_settings = dict(evaluation_settings)
        if (campaign_id == "g008-competitive-20261021-v2"
                and evaluation_settings.get("mode") == "validation"):
            packaged_opponent = ROOT / "training/eval/assets/g007-step-300.pt"
            mounted_opponent = Path(
                "/kaggle/input/g008-frozen-candidate/g007-step-300.pt")
            if file_sha256(mounted_opponent) != evaluation_settings.get("opponent_checkpoint_sha256"):
                raise RuntimeError("Mounted G007 step-300 dataset artifact differs from the frozen opponent hash.")
            packaged_opponent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mounted_opponent, packaged_opponent)
        evaluation_settings_sha256 = digest_json(evaluation_settings, "__no_digest_field__")
        evaluation_settings.setdefault("device", "cuda")
        report = evaluate_registered(evaluation_settings, output,
                                    config_file_sha256=file_sha256(ROOT / "run-config.json"),
                                    settings_sha256=evaluation_settings_sha256,
                                    runtime_evidence=runtime_evidence)
        print(f"Completed registered G008 {evaluation_settings['mode']} evaluation; "
              f"games={report['game_count']}; artifacts: {output}")
        return
    from training.ppo.train import TrainConfig, train
    config = TrainConfig(**settings["train_config"])
    if config.device != "cuda":
        raise RuntimeError("Kaggle training device must resolve to CUDA.")
    g008_train = settings.get("g008_train_schedule")
    if g008_train is not None:
        from training.eval.provenance import build_training_run_schedule, seed_plan_sha256
        schedule = build_training_run_schedule(
            run_id=g008_train["run_uuid"], root_seed=g008_train["root_seed"],
            arena_count=config.arena_count, total_updates=config.total_updates,
            rollout_steps=config.rollout_steps)
        if (schedule["expanded_schedule_digest"] != g008_train["expanded_schedule_digest"]
                or schedule["schedule_spec_digest"] != g008_train["schedule_spec_digest"]
                or seed_plan_sha256() != g008_train["seed_plan_sha256"]):
            raise RuntimeError("G008 training schedule differs from its frozen package binding.")
        train(config, output, export_onnx=True, run_evaluation=False,
              g008_schedule=schedule, run_id=g008_train["run_uuid"])
        runtime_evidence["run_id"] = g008_train["run_uuid"]
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed G008 train-only workload {run_id}; artifacts: {output}")
        return
    g008_campaign_train = settings.get("g008_campaign_train")
    if g008_campaign_train is not None:
        report = train(config, output, export_onnx=True, run_evaluation=False,
                       g008_campaign_train=g008_campaign_train,
                       run_id=g008_campaign_train["run_uuid"])
        runtime_evidence["run_id"] = g008_campaign_train["run_uuid"]
        (output / "kaggle-execution-evidence.json").write_text(
            json.dumps(runtime_evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"Completed registered G008 campaign train {run_id}; "
              f"updates={len(report['training']['updates'])}; artifacts: {output}")
        return
    train(config, output, export_onnx=True)
    benchmark_settings = settings.get("throughput_benchmark")
    if benchmark_settings is not None:
        from dataclasses import replace
        from training.ppo.benchmark import build_candidates, run_benchmark
        from training.ppo.protocol import digest_json, file_sha256
        benchmark_root = output / ".throughput-benchmark-work"
        base = TrainConfig(**settings["train_config"])
        candidates = [replace(candidate, total_updates=benchmark_settings["updates"])
                      for candidate in build_candidates(benchmark_settings["configs"], base)]
        if any(candidate.device != "cuda" for candidate in candidates):
            raise RuntimeError("Kaggle throughput benchmark must use the same CUDA device as training.")
        benchmark_report = run_benchmark(candidates, benchmark_root)
        report_path = output / "benchmark-results.json"
        report_path.write_text(json.dumps(benchmark_report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if benchmark_root.exists():
            shutil.rmtree(benchmark_root)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = report_path.relative_to(output).as_posix()
        manifest.setdefault("files", []).append({"path": relative, "sha256": file_sha256(report_path)})
        manifest.setdefault("benchmark_records", []).append({
            "protocol": benchmark_report["format"], "report_file": relative,
            "selected_trial": benchmark_report["selected_trial"],
            "selection_metric": benchmark_report["selection_metric"],
            "device_mode": benchmark_report["device_mode"],
            "execution_model": benchmark_report["execution_model"],
        })
        manifest["manifest_sha256"] = digest_json(manifest, "manifest_sha256")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Completed run {run_id}; artifacts: {output}")


if __name__ == "__main__":
    main()
