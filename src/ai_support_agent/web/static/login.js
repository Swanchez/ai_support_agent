async function sessionExists() {
  const response = await fetch("/api/v1/auth/session", { credentials: "same-origin" });
  return response.ok;
}

async function login(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const error = document.getElementById("login-error");
  const submit = form.querySelector("button");
  error.textContent = "";
  submit.disabled = true;
  try {
    const data = new FormData(form);
    const response = await fetch("/api/v1/auth/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login: data.get("login"), password: data.get("password") }),
    });
    if (!response.ok) throw new Error("Неверный логин или пароль.");
    window.location.replace("/");
  } catch (problem) {
    error.textContent = problem.message;
    submit.disabled = false;
  }
}

sessionExists().then((authenticated) => {
  if (authenticated) window.location.replace("/");
  else document.getElementById("login-form").addEventListener("submit", login);
});
