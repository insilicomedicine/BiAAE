from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, out_dim, dropout1d=0.2, dropout2d=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout2d(dropout2d),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout1d),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.conv(x).reshape(x.shape[0], -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.fc(x)
        return self.conv(x.reshape(x.shape[0], 16, 2, 2))


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.fc(x)


class BiAAE(ABC):
    def __init__(self, args):
        self.args = args
        self.submodels = self.create_submodels(args)
        self.optimizers = {
            name: torch.optim.Adam(submodel.parameters(),
                                   lr=self.args.learning_rate,
                                   weight_decay=self.args.l2)
            for name, submodel in self.submodels.items()}
        self.discriminator_submodels_names = ['discriminator']
        self.generator_submodels_names = [
            name for name in self.submodels.keys()
            if name not in self.discriminator_submodels_names]

    @staticmethod
    @abstractmethod
    def create_submodels(args):
        """
        Override this method to create necessary submodels dict with
        {'encoder_x', 'encoder_y', 'decoder_x',
         'decoder_y', 'discriminator'} keys
        based on the config in args
        """
        raise NotImplementedError

    def get_latent_variables(self, batch):
        zs_x = self.submodels['encoder_x'](batch['x'])
        # Dividing into exlcusive z_x and shared s_x part
        z_x, s_x = torch.split(zs_x, self.args.exclusive_x_dim, 1)

        zs_y = self.submodels['encoder_y'](batch['y'])
        # Dividing into exlcusive z_y and shared s_y part
        z_y, s_y = torch.split(zs_y, self.args.exclusive_y_dim, 1)

        if self.args.mode == 'UniAAE':
            s_x = s_y

        return z_x, s_x, z_y, s_y

    def generator_step(self, batch, backward=False, step=0, writer=None):
        (z_x, s_x, z_y, s_y) = self.get_latent_variables(batch)

        x_rec_x = self.submodels['decoder_x'](torch.cat((z_x, s_x), 1))
        # Reconstruction of x with shared part coming from y
        x_rec_y = self.submodels['decoder_x'](torch.cat((z_x, s_y), 1))

        y_rec_y = self.submodels['decoder_y'](torch.cat((z_y, s_y), 1))
        # Reconstruction of y with shared part coming from x
        y_rec_x = self.submodels['decoder_y'](torch.cat((z_y, s_x), 1))

        # Reconstruction losses
        loss_x_rec = (nn.MSELoss()(x_rec_x, batch['x']) +
                      nn.MSELoss()(x_rec_y, batch['x'])) / 2.
        loss_y_rec = (nn.MSELoss()(y_rec_x, batch['y']) +
                      nn.MSELoss()(y_rec_y, batch['y'])) / 2.

        # Loss for bringing shared parts together
        loss_shared = ((s_x - s_y) ** 2).mean()

        ones = torch.ones(2 * z_x.shape[0]).to(z_x.device)
        inputs = torch.cat(
            (torch.cat((z_x, s_x, z_y), 1), torch.cat((z_x, s_y, z_y), 1)),
            0)
        # Generator loss for imposing prior distribution, which also
        # ensures that three representations will be independent
        loss_generator = nn.BCEWithLogitsLoss()(
            self.submodels['discriminator'](inputs).reshape(-1),
            ones)

        loss = (loss_x_rec * self.args.loss_rec_lambda_x / 2 +
                loss_y_rec * self.args.loss_rec_lambda_y / 2 +
                loss_shared * self.args.loss_shared_lambda +
                loss_generator * self.args.loss_gen_lambda)
        loss_unweighted = (loss_x_rec + loss_y_rec +
                           loss_shared + loss_generator)

        if writer is not None:
            writer.add_scalar('generator/loss_reconstruction_x',
                              loss_x_rec.item(), step)
            writer.add_scalar('generator/loss_reconstruction_y',
                              loss_y_rec.item(), step)
            writer.add_scalar('generator/loss_reconstruction',
                              (loss_x_rec + loss_y_rec).item(), step)
            writer.add_scalar('generator/loss_shared',
                              loss_shared.item(), step)
            writer.add_scalar('generator/loss_generator',
                              loss_generator.item(), step)
            writer.add_scalar('generator/loss_unweighted',
                              loss_unweighted.item(), step)

            writer.add_histogram('generator/s_y-s_x', s_y - s_x, step)
            writer.add_histogram('generator/s_x', s_x, step)
            writer.add_histogram('generator/s_y', s_y, step)
            writer.add_histogram('generator/z_x', z_x, step)
            writer.add_histogram('generator/z_y', z_y, step)

        if backward:
            for name in self.generator_submodels_names:
                self.optimizers[name].zero_grad()
            loss.backward()
            for name in self.generator_submodels_names:
                self.optimizers[name].step()

    def discriminator_step(self, batch, backward=False, step=0, writer=None):
        (z_x, s_x, z_y, s_y) = (
            t.detach()
            for t in self.get_latent_variables(batch)
        )

        device = z_x.device
        n = z_x.shape[0]  # batch size

        input_xy = torch.cat((z_x, s_x, z_y), 1)
        input_yx = torch.cat((z_x, s_y, z_y), 1)

        inputs = torch.cat((input_xy, input_yx), 0)
        noise = torch.randn_like(inputs)

        ones = torch.ones(2 * n).to(device)
        zeros = torch.zeros(2 * n).to(device)

        # Compare <z_x, s_x, z_y> or <z_x, s_y, z_y> vs N(0, I),
        outputs = self.submodels['discriminator'](
            torch.cat((inputs, noise), 0)
        ).reshape(-1)
        loss = nn.BCEWithLogitsLoss()(outputs, torch.cat((zeros, ones), 0))

        if writer is not None:
            writer.add_scalar('discriminator/loss', loss.item(), step)

        if backward:
            for name in self.discriminator_submodels_names:
                self.optimizers[name].zero_grad()
            loss.backward()
            for name in self.discriminator_submodels_names:
                self.optimizers[name].step()

    def step(self, batch, backward=False, step=0, writer=None):
        batch = {'x': batch[0], 'y': batch[1]}
        n = self.args.discriminator_steps
        for i in range(n):
            self.discriminator_step(
                batch, backward=backward, step=step,
                writer=writer if i == n - 1 else None)
        n = self.args.generator_steps
        for i in range(n):
            self.generator_step(
                batch, backward=backward, step=step,
                writer=writer if i == n - 1 else None)

    @staticmethod
    def generate_step(condition, exclusive_object_dim, exclusive_condition_dim,
                      encoder, decoder):
        noise = torch.randn(condition.shape[0],
                            exclusive_object_dim).to(condition.device)
        shared = encoder(condition)[:, exclusive_condition_dim:]
        return decoder(torch.cat((noise, shared), 1))

    def generate(self, condition, batch_size=512, x_as_condition=False):
        with torch.no_grad():
            if x_as_condition:
                exclusive_object_dim = self.args.exclusive_y_dim
                exclusive_condition_dim = self.args.exclusive_x_dim
                encoder = self.submodels['encoder_x']
                decoder = self.submodels['decoder_y']
            else:
                exclusive_object_dim = self.args.exclusive_x_dim
                exclusive_condition_dim = self.args.exclusive_y_dim
                encoder = self.submodels['encoder_y']
                decoder = self.submodels['decoder_x']

            batch_size = min(batch_size, condition.shape[0])
            generated = []
            for indices in torch.split(torch.arange(condition.shape[0]),
                                       batch_size):
                generated.append(BiAAE.generate_step(
                    condition[indices], exclusive_object_dim,
                    exclusive_condition_dim, encoder, decoder))
            return torch.cat(generated, 0)


class ConvolutionalBiAAE(BiAAE):
    def __init__(self, *args, **kwargs):
        BiAAE.__init__(self, *args, **kwargs)

    @staticmethod
    def create_submodels(args):
        submodels = {'encoder_x': Encoder(
                        args.exclusive_x_dim + args.shared_dim,
                        dropout1d=args.dropout1d, dropout2d=args.dropout2d
                     ),
                     'decoder_x': Decoder(
                         args.exclusive_x_dim + args.shared_dim
                     ),
                     'encoder_y': Encoder(
                         args.exclusive_y_dim + args.shared_dim,
                         dropout1d=args.dropout1d, dropout2d=args.dropout2d
                     ),
                     'decoder_y': Decoder(
                         args.exclusive_y_dim + args.shared_dim
                     ),
                     'discriminator': Discriminator(
                         args.exclusive_x_dim + args.shared_dim +
                         args.exclusive_y_dim
                     )}
        return submodels
