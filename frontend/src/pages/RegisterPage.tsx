import { useState } from "react"
import { useMutation } from "@tanstack/react-query"
import { useNavigate, Link } from "react-router-dom"
import { register } from "@/api/auth"
import { useAuthStore } from "@/store/authStore"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function RegisterPage() {
  const [username, setUsername] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const { setAuth } = useAuthStore()
  const navigate = useNavigate()

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => register({ username, email, password }),
    onSuccess: (data) => {
      setAuth(data.user, data.access_token)
      navigate("/feed")
    },
  })

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <div className="w-full max-w-md p-8 bg-slate-900 rounded-2xl space-y-4">
        <h1 className="text-2xl font-bold text-white">Create account</h1>
        <Input placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} />
        <Input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
        <Input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} />
        {error && <p className="text-red-400 text-sm">Registration failed. Please try again.</p>}
        <Button className="w-full" onClick={() => mutate()} disabled={isPending}>
          {isPending ? "Creating account…" : "Create account"}
        </Button>
        <p className="text-center text-sm text-slate-400">
          Already have an account?{" "}
          <Link to="/login" className="underline text-white">Sign in</Link>
        </p>
      </div>
    </div>
  )
}
