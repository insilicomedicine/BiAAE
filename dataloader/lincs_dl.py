import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset as TorchDataset
from random import randint
import pandas as pd
from sklearn.model_selection import train_test_split
import re


def parse_string_lists(data):
    for c in ['ControlGeneExps', 'TumourGeneExps']:
        if c in data.columns:
            data[c] = data[c].str.split()
    for c in ['ControlIndices', 'TumourIndices']:
        if c in data.columns:
            data[c] = data[c].str.split().map(lambda x: list(map(int, x)))


def convert_indices(series_of_lists):
    ''' converts all values-indices of series of lists into new indices, corresponding only to existing values
        returns unique indices for further slicing big array and new transformed series of lists
    '''
    initial_indices = unroll_column(series_of_lists)
    indices2new_indices = {k: i for i, k in enumerate(initial_indices)}
    new_series = series_of_lists.map(
        lambda ls: [indices2new_indices[l] for l in ls])
    return initial_indices, new_series


class LincsDataSet:
    def __init__(self, data_dir='mols', seed=0,
                 cell_line=None, ptime=None,
                 use_diff_expression=False,
                 only_transcriptomes=False,
                 experimentsfile='experiments_filtered.csv'):
        r"""
        :param data_dir: data directory with controls, tumors, and experiments csv file
        :param gpu: boolean, whether to save genes and molecules in GPU Tensor or numpy array.
        :param cell_line: keep in memory only this cell line for efficiency reasons
        :param ptime: keep in memory only this perturbation time for efficiency reasons
        :param use_diff_expression: use difference of cell after - cell before instead of cell before absolute values
        :param only_transcriptomes: if True, does not load molecules. Useful for knockdowns or overexpressions
        :param experiment_gene: If True, dataset has corresponding gene for each experiment. Useful for knockdowns\overexpressions
        """
        self.seed = seed
        self.only_transcriptomes = only_transcriptomes
        self.use_diff_expression = use_diff_expression
        controlfile = os.path.join(data_dir, 'robust_normalized_controls.npz')
        tumorfile = os.path.join(data_dir, 'robust_normalized_tumors.npz')

        self.npz_controls = np.load(controlfile, allow_pickle=True)
        self.npz_tumors = np.load(tumorfile, allow_pickle=True)
        self._load_genes()

        self.experimentsfile = os.path.join(data_dir, experimentsfile)
        self._load_experiments()

        if cell_line is not None or ptime is not None:
            if cell_line is not None:
                assert cell_line in self.experiments.CellLine.unique()
                self.experiments = self.experiments[
                    self.experiments.CellLine == cell_line]
            if ptime is not None:
                assert ptime in self.experiments.Time.unique()
                self.experiments = self.experiments[
                    self.experiments.Time == ptime]
            self.indices_to_slice, self.experiments[
                'TumourIndices'] = convert_indices(
                self.experiments['TumourIndices'])
            self.tumor_genes = self.tumor_genes[self.indices_to_slice]
            self.indices_to_slice, self.experiments[
                'ControlIndices'] = convert_indices(
                self.experiments['ControlIndices'])
            self.control_genes = self.control_genes[self.indices_to_slice]

    def _load_genes(self):
        self.control_genes = torch.from_numpy(
            self.npz_controls['genes'].astype('float32'))
        self.tumor_genes = torch.from_numpy(
            self.npz_tumors['genes'].astype('float32'))
        self.gene_scaler = self.npz_controls['scaler']
        self.control_distils = self.npz_controls['gene_ids']
        self.tumor_distils = self.npz_tumors['gene_ids']
        self.control_gene_names = self.npz_controls['column_names']
        self.tumor_gene_names = self.npz_tumors['column_names']
        assert (self.control_gene_names == self.tumor_gene_names).all()

    def _load_experiments(self):
        self.experiments = pd.read_csv(self.experimentsfile)
        parse_string_lists(self.experiments)
        if not self.only_transcriptomes:
            self.experiments['DrugIndex'] = self.experiments[
                'DrugIndex_maccs']


class LincsSampler(TorchDataset):
    def transform_dose(self, dose_array):
        transformed_dose = torch.from_numpy(np.log1p(dose_array) / np.log(11.))[
                           :,
                           None].float()  # logarithmic scale, s.t. 10 micromoles -> 1 unit
        return transformed_dose

    def transform_ptime(self, ptime_array):
        transformed_ptime = torch.from_numpy(ptime_array)[:,
                            None].float() / 24.  # Time in 24 hour units
        return transformed_ptime

    def transform_tissue(self, tissue_array):
        transformed_tissue = torch.from_numpy(
            tissue_array)  # One-hot coded Tissue types
        return transformed_tissue

    def __init__(self, lincs_dataset, test_set=None, include_drugIDs=[],
                 exclude_drugIDs=[],
                 cell_line=None, use_smiles=False, ptime=None, dose=None,
                 reverse=False):
        r'''
        :param lincs_dataset: common LincsDataSet for train\validation\test samplers
        :param test_set: If test_set is None: use all the data. test_set == False or 0 corresponds to the training set,
            test_set == True or 1 corresponds to the validation set, test_set == 2 corresponds to the test set
        :param include_drugIDs: include these drugs
        :param exclude_drugIDs: exclude these drugs
        :param cell_line: consider only this cell line
        :param ptime: consider only this perturbation time
        :param dose: consider only this dose
        '''
        self.dataset = lincs_dataset
        self.control_genes = self.dataset.control_genes
        self.tumor_genes = self.dataset.tumor_genes
        self.use_diff_expression = self.dataset.use_diff_expression
        self.include_drugIDs = include_drugIDs
        self.exclude_drugIDs = exclude_drugIDs
        self.sample_negative = False
        self.use_smiles = use_smiles
        np.random.seed(self.dataset.seed)
        self.experiments = self.dataset.experiments.copy()

        self.reverse = reverse

        if not self.dataset.only_transcriptomes:
            self.mol_ids = self.experiments.DrugIndex.unique()

            if test_set is not None:  # train or validation or test set
                np.random.shuffle(self.mol_ids)
                division_point = self.mol_ids.shape[0] // 4
                if not test_set:  # train set
                    self.mol_ids = self.mol_ids[2 * division_point:]
                elif test_set > 1:  # test set
                    self.mol_ids = self.mol_ids[
                                   division_point:2 * division_point]
                else:  # validation set
                    self.mol_ids = self.mol_ids[:division_point]
                self.mol_ids = np.unique(
                    np.concatenate((self.mol_ids, self.include_drugIDs)))
            self.mol_ids = self.mol_ids[
                ~np.isin(self.mol_ids, self.exclude_drugIDs)]

            self.experiments = self.experiments[
                self.experiments.DrugIndex.isin(self.mol_ids)].copy()
            if dose is not None:
                self.experiments = self.experiments[
                    self.experiments.Dose == dose]

        if cell_line is not None:
            self.experiments = self.experiments[
                self.experiments.CellLine == cell_line]
        if ptime is not None:
            self.experiments = self.experiments[self.experiments.Time == ptime]

        self.Time = self.transform_ptime(self.experiments.Time.values)
        self.Tissue = self.transform_tissue(self.experiments[[c for c in
                                                              self.experiments.columns
                                                              if c.startswith(
                'primary_site')]].values)
        self.CellBeforeIndices = self.experiments.ControlIndices.values
        self.CellAfterIndices = self.experiments.TumourIndices.values
        self.smiles = self.experiments.SMILES.values

        self.DrugIndex = torch.from_numpy(self.experiments.DrugIndex.values)
        self.Dose = self.transform_dose(self.experiments.Dose.values)

        self.empty = torch.zeros(1)
        self.N_for_negative = self.experiments.shape[0] - 1

    def __len__(self):
        return self.experiments.shape[0]

    def __getitem__(self, idx):
        before = self.CellBeforeIndices[idx]
        after = self.CellAfterIndices[idx]
        before = self.control_genes[before[randint(0, len(before) - 1)]]
        after = self.tumor_genes[after[randint(0, len(after) - 1)]]
        if self.use_diff_expression:
            before = after - before

        if self.reverse:
            return torch.cat(
                (after - before, self.Dose[idx]),
                dim=-1), self.smiles[idx]
        else:
            return self.smiles[idx], torch.cat(
                (after - before, self.Dose[idx]), dim=-1)
