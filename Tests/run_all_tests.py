import unittest

# Importă toate clasele de test din fișierele tale
from test_window import TestMainWindowV2
from test_settings_window import TestSettingsWindow
from test_repo import TestAudioRepository
from test_entity import TestEntities
from test_audio_service import TestAudioService
from test_audio_processor import TestAudioProcessor
from test_audio_device_manager import TestAudioDeviceManagerComplete

def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Adaugă fiecare clasă de testare în suită
    suite.addTests(loader.loadTestsFromTestCase(TestMainWindowV2))
    suite.addTests(loader.loadTestsFromTestCase(TestSettingsWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioRepository))
    suite.addTests(loader.loadTestsFromTestCase(TestEntities))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioService))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioDeviceManagerComplete))

    # Rulează testele
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Cod de ieșire convențional: 0 = OK, 1 = Failed
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)

if __name__ == "__main__":
    run_all()
