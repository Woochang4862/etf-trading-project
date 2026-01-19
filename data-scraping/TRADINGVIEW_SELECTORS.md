# TradingView Automation Selectors (2024-2025)

Based on exhaustive analysis of the codebase and external resources, these are the most reliable selectors for TradingView automation.

## 1. Authentication Elements

### Login Button (Home Page)
- **Primary (CSS):** `button[aria-label="Open user menu"]` (User Icon)
- **Secondary (CSS):** `button:has-text("Sign in")` (or localized: `"로그인"`)
- **Strategy:** Look for the user menu icon or the specific text.

### Email Login Option
- **Primary (CSS):** `button[name="Email"]`
- **Secondary (CSS):** `button:has-text("Email")` (or localized: `"이메일"`)
- **XPath:** `//span[contains(text(), 'Email')]/ancestor::button`

### Username Input
- **Primary (CSS):** `input[name="username"]`
- **Secondary (CSS):** `input[type="text"][name="username"]`
- **Fallback (CSS):** `input[placeholder*="username"]` (or localized: `"유저네임"`)
- **Note:** `name="username"` is highly stable across updates.

### Password Input
- **Primary (CSS):** `input[name="password"]`
- **Secondary (CSS):** `input[type="password"]`
- **Note:** `type="password"` is extremely reliable as it's required for browser security handling.

### Submit Button
- **Primary (CSS):** `button[type="submit"]`
- **Secondary (CSS):** `button:has-text("Sign in")` (or localized: `"로그인"`)
- **XPath:** `//button[@type='submit']`

## 2. Dynamic Element Strategy

TradingView frequently randomizes class names (e.g., `tv-signin-dialog__input`, `js-header__signin`). **Avoid using these.**

**Recommended Strategy:**
1.  **Attribute Matching:** Prioritize stable attributes like `name`, `type`, `aria-label`, `data-name`, or `data-test` (if available).
2.  **Text Content:** Use Playwright's `get_by_text()` or Selenium's text matching for buttons, as these are user-facing and change less often than internal IDs.
3.  **Container Scoping:** Locate the modal dialog first (e.g., `div[data-dialog-name="gopro"]`) to limit the search scope.

## 3. Playwright Implementation Example

```python
# Click User Icon / Sign in
page.locator('button[aria-label="Open user menu"]').click()
# OR
page.get_by_text("Sign in").click()

# Select Email Method
page.get_by_role("button", name="Email").click()

# Fill Credentials
page.locator('input[name="username"]').fill("your_username")
page.locator('input[name="password"]').fill("your_password")

# Submit
page.locator('button[type="submit"]').click()
```
