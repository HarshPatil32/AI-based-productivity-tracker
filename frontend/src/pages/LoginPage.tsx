import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { useNavigate, Link } from "react-router-dom"
import { login } from "@/api/auth"
import { useAuthStore } from "@/store/authStore"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [rememberMe, setRememberMe] = useState(false)
  const { setAuth } = useAuthStore()
  const navigate = useNavigate()

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => login({ email, password }),
    onSuccess: (data) => {
      setAuth(data.user, data.access_token)
      navigate("/feed")
    },
  })

  return (
    <div className="h-screen w-screen flex overflow-hidden">
      {/* Left Panel */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-10 bg-gradient-to-br from-blue-600 to-indigo-700 text-white">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full border-2 border-white flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <circle cx="12" cy="12" r="9" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 3" />
            </svg>
          </div>
          <span className="text-lg font-semibold">FocusTrack</span>
        </div>

        {/* Hero Text */}
        <div className="space-y-8">
          <div className="space-y-3">
            <h1 className="text-4xl font-bold leading-tight">
              Track Your Focus.<br />Share Your Progress.
            </h1>
            <p className="text-blue-100 text-base">
              AI-powered attention monitoring meets social productivity.
            </p>
          </div>

          {/* Features */}
          <div className="space-y-5">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-blue-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <div>
                <p className="font-semibold">AI Attention Tracking</p>
                <p className="text-sm text-blue-200">Real-time focus monitoring with computer vision</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-blue-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <div>
                <p className="font-semibold">Social Progress</p>
                <p className="text-sm text-blue-200">Share sessions and compete with friends</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-blue-200" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="9" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 7v5l3 3" />
                </svg>
              </div>
              <div>
                <p className="font-semibold">Detailed Analytics</p>
                <p className="text-sm text-blue-200">Track productivity trends over time</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <p className="text-sm text-blue-200">© 2026 FocusTrack. Privacy-first productivity.</p>
      </div>

      {/* Right Panel */}
      <div className="flex-1 flex items-center justify-center bg-slate-100 p-8 relative h-full">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-md p-10 space-y-6">
          <div className="text-center space-y-1">
            <h2 className="text-2xl font-bold text-gray-900">Welcome Back</h2>
            <p className="text-sm text-gray-500">Sign in to continue tracking</p>
          </div>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-gray-700">Email</label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="bg-white border-gray-300 text-gray-900 placeholder:text-gray-400"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium text-gray-700">Password</label>
              <Input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="bg-white border-gray-300 text-gray-900"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={e => setRememberMe(e.target.checked)}
                  className="w-4 h-4 accent-blue-600"
                />
                Remember me
              </label>
              <button className="text-sm text-blue-600 hover:underline">Forgot password?</button>
            </div>

            {error && <p className="text-red-500 text-sm">Invalid credentials</p>}

            <Button
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold"
              onClick={() => mutate()}
              disabled={isPending}
            >
              {isPending ? "Signing in…" : "Sign In"}
            </Button>
          </div>

          <p className="text-center text-sm text-gray-500">
            Don't have an account?{" "}
            <Link to="/register" className="text-blue-600 font-semibold hover:underline">Sign up</Link>
          </p>
        </div>

        {/* Help button */}
        <button className="absolute bottom-6 right-6 w-9 h-9 rounded-full bg-gray-800 text-white flex items-center justify-center text-sm font-bold shadow hover:bg-gray-700">
          ?
        </button>
      </div>
    </div>
  )
}
