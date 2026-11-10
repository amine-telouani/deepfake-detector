import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

STREAMLIT_URL = "https://amine-telouani-deepfake-detector.streamlit.app/"

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(STREAMLIT_URL)
        print(f"Opened: {STREAMLIT_URL}")

        try:
            wake_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'get this app back up')]"))
            )
            wake_button.click()
            print("Wake-up button clicked ✅")
            time.sleep(10)
        except TimeoutException:
            print("No wake-up button found. App is likely already awake ✅")

    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    finally:
        driver.quit()
        print("Script finished.")

if __name__ == "__main__":
    main()