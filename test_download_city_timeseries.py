"""Offline checks: python -m unittest test_download_city_timeseries -v"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio as rio

import vlm.download_city_timeseries as download


class DownloaderTests(unittest.TestCase):
    def test_exact_grid_and_season(self):
        grid = download.make_grid(35.236864, 52.336017, 10)
        self.assertEqual(grid['epsg'], 32639)
        self.assertEqual((grid['width'], grid['height']), (1000, 1000))
        left, bottom, right, top = grid['bounds']
        self.assertEqual((right - left, top - bottom), (10000, 10000))
        self.assertEqual(download.season(2020, 6, 9), ('2020-06-01', '2020-10-01'))
        self.assertEqual(download.season(2020, 1, 12), ('2020-01-01', '2021-01-01'))
        with self.assertRaises(ValueError):
            download.make_grid(35, 52, 10.001)

    def test_chunk_assembly_including_partial_edges_and_nodata(self):
        grid = download.make_grid(35.236864, 52.336017, 5.01)

        def fake_chunk(image, grid, row, col, height, width):
            arr = np.full((12, height, width), 0.2, dtype='float32')
            arr[11] = 7
            arr[0] = row + col
            if row == col == 0:
                arr[:11, 0, 0] = download.NODATA
                arr[11, 0, 0] = 0
            return arr

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with patch.object(download, 'download_chunk', side_effect=fake_chunk) as mock:
                coverage = download.save_year(None, grid, folder)
            self.assertEqual(mock.call_count, 4)
            self.assertGreater(coverage, 99)
            with rio.open(folder / 'reflectance.tif') as ref, rio.open(folder / 'clear_observations.tif') as qa:
                self.assertEqual(ref.transform, qa.transform)
                self.assertEqual(ref.crs.to_epsg(), 32639)
                self.assertEqual(ref.res, (10, 10))
                self.assertEqual(ref.descriptions, tuple(download.BANDS))
                self.assertEqual(ref.read(1)[500, 500], 1000)
                self.assertTrue(ref.read(1, masked=True).mask[0, 0])
                self.assertEqual(qa.read(1)[500, 500], 7)
                self.assertEqual(qa.read(1)[0, 0], 0)
            download.preview(folder)
            self.assertTrue((folder / 'rgb.png').exists())


if __name__ == '__main__':
    unittest.main()
